"""RLOO + WM and RLB + CWM: one on-policy simulator rollout per group, the rest from the world model.

RLB + CWM reuses the DDIM-inverted noise of the factual transition for every group member;
RLOO + WM gives each member fresh noise. Generated members are rewarded by the frozen
reward classifier, the factual member by the simulator.
"""

import argparse
import copy
import os
import random
import time
from collections import deque

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from PIL import Image

from common import DEVICE, DINO_RL_DIR, Episode, envs
from action_repeat import make_repeat_pixel_envs
from diffusion.config import BUFFER_SIZE, HEIGHT, WIDTH
from diffusion.data import IMG_TRANSFORMS
from diffusion.reward_rules import DinoRewardRule
from estimators import leave_one_out_advantages, leave_one_out_loss
from policy_core import K, make_policy, stack_to_tensor
from policy_roots import (
    advance_to_obstacle,
    build_root_seeds,
    parse_root_speed_bins,
    parse_root_ttc_bins,
)
from train_policy import auto_validate_checkpoint
from world_model import (
    DTYPES,
    WorldModel,
    load_reward_model,
    policy_frame,
    reward_model_probs,
)


def collect_world_model_groups(
    world,
    reward_model,
    policy,
    rng,
    num_groups,
    group_size,
    decisions,
    method,
    collision_penalty,
    action_repeat,
    root_ttc_bins,
    root_speed_bins,
    keep_frames=False,
):
    """Collect one factual and group_size-1 generated trajectories per group."""
    if group_size < 2:
        raise ValueError("model-assisted methods require group_size >= 2")
    if method not in {"rlb_cwm", "rloo_wm"}:
        raise ValueError(f"unknown policy method {method}")

    seeds, ttcs, _, speed_targets = build_root_seeds(
        rng,
        num_groups,
        1,
        True,
        root_ttc_bins=root_ttc_bins,
        root_speed_bins=root_speed_bins,
    )
    factual_envs, _ = make_repeat_pixel_envs(
        num_groups, game_mode="train", k=action_repeat
    )

    factual_eps = [Episode() for _ in range(num_groups)]
    factual_actions = np.zeros((num_groups, decisions), dtype=np.int64)
    factual_rewards = np.zeros((num_groups, decisions), dtype=np.float32)
    factual_frames = []
    policy_stacks = []
    history = (BUFFER_SIZE - 1) * action_repeat

    # Factual rollouts in the simulator; the last 8 decision frames before the root are the context.
    try:
        for g, simulator in enumerate(factual_envs):
            root, recent = advance_to_obstacle(
                simulator,
                seeds[g],
                history=history,
                target_ttc=ttcs[g],
                target_speed=speed_targets[g],
            )
            raw_frames = list(recent) + [root]
            last = len(raw_frames) - 1
            context = [
                raw_frames[max(last - offset * action_repeat, 0)]
                for offset in range(BUFFER_SIZE - 1, -1, -1)
            ]
            factual_frames.append(list(context))
            gray_context = [np.asarray(frame)[..., 0] for frame in context]
            policy_stacks.append(
                deque([policy_frame(frame) for frame in gray_context[-K:]], maxlen=K)
            )

        for t in range(decisions):
            states = torch.stack([stack_to_tensor(stack) for stack in policy_stacks])
            dist = torch.distributions.Categorical(logits=policy(states))
            actions = dist.sample()
            log_probs, entropies = dist.log_prob(actions), dist.entropy()
            for g, simulator in enumerate(factual_envs):
                next_frame, reward, terminated, *_ = simulator.step(
                    envs.Action(int(actions[g].item()))
                )
                if terminated:
                    raise RuntimeError(
                        "factual simulator terminated inside a fixed-length train-mode rollout"
                    )
                reward = float(reward)
                if reward < 0:
                    reward *= abs(collision_penalty)
                factual_eps[g].add(states[g], log_probs[g], entropies[g], reward)
                factual_actions[g, t] = int(actions[g].item())
                factual_rewards[g, t] = reward
                factual_frames[g].append(next_frame)
                policy_stacks[g].append(policy_frame(np.asarray(next_frame)[..., 0]))
    finally:
        for simulator in factual_envs:
            simulator.close()

    # Encode the factual frames; RLB + CWM also needs the targets for inversion.
    all_images = torch.stack(
        [
            torch.stack(
                [
                    IMG_TRANSFORMS(Image.fromarray(frame).convert("RGB"))
                    for frame in frames
                ]
            )
            for frames in factual_frames
        ]
    )
    G = all_images.shape[0]
    context_lat = world.encode(
        all_images[:, :BUFFER_SIZE].flatten(0, 1), sample=True
    ).view(G, BUFFER_SIZE, -1, HEIGHT // 8, WIDTH // 8)
    real_lat = context_lat
    if method == "rlb_cwm":
        target_lat = world.encode(
            all_images[:, BUFFER_SIZE:].flatten(0, 1), sample=False
        ).view(G, decisions, -1, HEIGHT // 8, WIDTH // 8)
        real_lat = torch.cat([context_lat, target_lat], dim=1)

    factual_gray = np.stack(
        [
            np.stack(
                [
                    np.asarray(
                        Image.fromarray(np.asarray(frame)[..., 0]).resize(
                            (WIDTH, HEIGHT), Image.Resampling.BILINEAR
                        )
                    )
                    for frame in frames
                ]
            )
            for frames in factual_frames
        ]
    )
    factual_tokens = torch.as_tensor(factual_actions, device=DEVICE)

    # Rule-based reward on the same frames, logged as a diagnostic only.
    warm_rules = []
    for g in range(G):
        rule = DinoRewardRule(max_frames=256)
        rule.reset(factual_gray[g, : BUFFER_SIZE - 1], [0] * (BUFFER_SIZE - 1))
        warm_rules.append(rule)

    def clone_rule(g):
        return copy.deepcopy(
            warm_rules[g],
            {
                id(warm_rules[g].cum_candidates): warm_rules[g].cum_candidates,
                id(warm_rules[g].speed_candidates): warm_rules[g].speed_candidates,
            },
        )

    # Generated members start from their group's factual context.
    c = group_size - 1
    n_cf = G * c
    member_group = np.repeat(np.arange(G), c)
    member_group_t = torch.as_tensor(member_group, device=DEVICE)
    cf_lat = real_lat[member_group_t, :BUFFER_SIZE].clone()
    cf_stacks = [
        deque(
            [
                policy_frame(factual_gray[g, j])
                for j in range(BUFFER_SIZE - K, BUFFER_SIZE)
            ],
            maxlen=K,
        )
        for g in member_group
    ]
    cf_eps = [Episode() for _ in range(n_cf)]
    generated_gray = factual_gray[member_group, BUFFER_SIZE - 1].copy()
    cf_actions = np.zeros((n_cf, decisions), dtype=np.int64)
    cf_rewards = np.zeros((n_cf, decisions), dtype=np.float32)
    cf_crashes = np.zeros(n_cf, dtype=np.float32)
    cf_passes = np.zeros(n_cf, dtype=np.float32)
    rule_gen_crashes = np.zeros(n_cf, dtype=np.float32)
    rule_gen_passes = np.zeros(n_cf, dtype=np.float32)
    generated_crash_agreement = np.zeros(n_cf, dtype=np.float32)
    reference_crashes = np.zeros(n_cf, dtype=np.float32)
    reference_passes = np.zeros(n_cf, dtype=np.float32)
    generated_rules = [clone_rule(g) for g in member_group]
    reference_rules = [clone_rule(g) for g in member_group]
    shown = [generated_gray.copy()] if keep_frames else None

    for t in range(decisions):
        states = torch.stack([stack_to_tensor(stack) for stack in cf_stacks])
        dist = torch.distributions.Categorical(logits=policy(states))
        actions = dist.sample()
        log_probs, entropies = dist.log_prob(actions), dist.entropy()
        cf_actions[:, t] = actions.detach().cpu().numpy()

        rule_dones = np.zeros(n_cf, dtype=bool)
        for i, action in enumerate(cf_actions[:, t]):
            reward, done = generated_rules[i].step(generated_gray[i], int(action))
            rule_gen_crashes[i] += done
            rule_gen_passes[i] += reward + done
            rule_dones[i] = bool(done)
            reward, done = reference_rules[i].step(
                factual_gray[member_group[i], BUFFER_SIZE - 1 + t], int(action)
            )
            reference_crashes[i] += done
            reference_passes[i] += reward + done

        if method == "rlb_cwm":
            # Shared noise: invert the factual transition once per group.
            noise_by_group = world.abduct(
                real_lat[:, BUFFER_SIZE + t],
                real_lat[:, t : t + BUFFER_SIZE],
                factual_tokens[:, t].unsqueeze(1),
            )
            noise = noise_by_group[member_group_t]
        else:
            noise = (
                torch.randn(
                    (n_cf, *context_lat.shape[2:]),
                    device=DEVICE,
                    dtype=context_lat.dtype,
                )
                * world.scheduler.init_noise_sigma
            )
        latent = world.predict(noise, cf_lat, actions.unsqueeze(1))
        cf_lat = torch.cat([cf_lat[:, 1:], latent.unsqueeze(1)], dim=1)
        next_gray = world.decode_gray(latent)

        probs = reward_model_probs(reward_model, generated_gray, next_gray)
        classes = probs.argmax(1)  # 0 crash, 1 neutral, 2 pass
        rewards = classes.astype(np.float32) - 1.0
        cf_crashes += classes == 0
        cf_passes += classes == 2
        generated_crash_agreement += (classes == 0) == rule_dones
        for i in range(n_cf):
            reward = float(rewards[i])
            if reward < 0:
                reward *= abs(collision_penalty)
            cf_eps[i].add(states[i], log_probs[i], entropies[i], reward)
            cf_rewards[i, t] = reward
            if t < decisions - 1:
                cf_stacks[i].append(policy_frame(next_gray[i]))
        generated_gray = next_gray
        if keep_frames:
            shown.append(generated_gray.copy())

    # Group-major order [factual, member, member, ...] for the leave-one-out baseline.
    episodes = []
    factual_mask = []
    for g in range(G):
        episodes.append(factual_eps[g])
        factual_mask.append(True)
        for i in np.flatnonzero(member_group == g):
            episodes.append(cf_eps[i])
            factual_mask.append(False)

    exact_crashes = (factual_rewards < 0).sum(1)
    exact_passes = (factual_rewards > 0).sum(1)
    stats = {
        "factual_crashes": exact_crashes,
        "factual_passes": exact_passes,
        "cf_crashes": cf_crashes,
        "cf_passes": cf_passes,
        "rule_gen_crashes": rule_gen_crashes,
        "rule_gen_passes": rule_gen_passes,
        "reference_crashes": reference_crashes,
        "reference_passes": reference_passes,
        "collision_agreement": generated_crash_agreement / decisions,
        "real_interactions": G * decisions,
        "model_interactions": n_cf * decisions,
        "factual_mask": np.asarray(factual_mask),
    }
    frames = None
    if keep_frames:
        frames = {
            "factual": factual_gray,
            "generated": np.stack(shown, axis=1),
            "factual_actions": factual_actions,
            "cf_actions": cf_actions,
            "factual_rewards": factual_rewards,
            "cf_rewards": cf_rewards,
            "member_group": member_group,
        }
    return episodes, stats, frames


def save_rollout_grid(path, frames, groups=2):
    from PIL import ImageDraw

    scale = (160, 120)
    T = frames["generated"].shape[1]
    sheets = []
    for g in range(min(groups, len(frames["factual"]))):
        members = np.flatnonzero(frames["member_group"] == g)
        sheet = Image.new("L", (T * scale[0], (len(members) + 1) * scale[1]), 255)
        draw = ImageDraw.Draw(sheet)
        for t in range(T):
            factual = frames["factual"][g, BUFFER_SIZE - 1 + t]
            sheet.paste(Image.fromarray(factual).resize(scale), (t * scale[0], 0))
        draw.text((4, 2), "simulator factual", fill=0)
        for row, i in enumerate(members, 1):
            for t in range(T):
                image = frames["generated"][i, t]
                sheet.paste(
                    Image.fromarray(image).resize(scale), (t * scale[0], row * scale[1])
                )
                if t < frames["cf_actions"].shape[1]:
                    draw.text(
                        (t * scale[0] + 4, row * scale[1] + 2),
                        f"a={frames['cf_actions'][i, t]} r={frames['cf_rewards'][i, t]:+.0f}",
                        fill=0,
                    )
        sheets.append(sheet)
    grid = Image.new("L", (sheets[0].width, sum(s.height for s in sheets)), 255)
    y = 0
    for sheet in sheets:
        grid.paste(sheet, (0, y))
        y += sheet.height
    grid.save(path)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(args.seed)

    root_ttc_bins = parse_root_ttc_bins(args.root_ttc_bins)
    root_speed_bins = parse_root_speed_bins(args.root_speed_bins)

    envs.BIRD_SPAWN_PROB = args.bird_spawn_prob
    envs.BIRD_MIN_SPEED = args.bird_min_speed
    low, mid, high = envs.BIRD_Y_POSITIONS[:3]
    envs.BIRD_Y_POSITIONS = (low, *([mid] * args.mid_bird_weight), high)

    output_root = args.output_root or os.path.join(
        DINO_RL_DIR, "runs", "policies", "checkpoints"
    )
    validation_log_dir = args.validation_log_dir or os.path.join(
        os.path.dirname(output_root), "validation"
    )
    out_dir = os.path.join(output_root, args.results_dir)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise RuntimeError(
            f"refusing to overwrite nonempty results directory {out_dir}"
        )
    os.makedirs(out_dir, exist_ok=True)

    batch = args.num_groups * args.group_size
    method_name = "RLB + CWM" if args.method == "rlb_cwm" else "RLOO + WM"
    generated_name = (
        "CWM counterfactuals"
        if args.method == "rlb_cwm"
        else "fresh-noise WM rollouts"
    )
    print(
        f"=== {method_name}: {args.num_groups} groups x {args.group_size} "
        f"= {batch} rollouts, H={args.decisions}, seed={args.seed} ==="
    )
    print(
        "per group: 1 current-policy simulator factual + "
        f"{args.group_size - 1} current-policy {generated_name}"
    )
    print(
        "rewards: exact simulator reward for factual; frozen visual classifier for generated members"
    )
    print("gradient: score functions from all factual and generated members")
    ttc_label = ", ".join(f"({lo},{hi}]" for lo, hi in root_ttc_bins)
    print(f"roots: fully standing, TTC {ttc_label}, speed bins {root_speed_bins}")

    world = WorldModel(
        args.model_folder,
        args.num_inference_steps,
        args.cfg_scale,
        args.dtype,
        args.vae_chunk,
    )
    reward_model = load_reward_model(args.reward_model)
    policy = make_policy(K, 3, orthogonal=True).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    hist = {
        key: []
        for key in (
            "reward",
            "entropy",
            "gradnorm",
            "group_spread",
            "factual_crashes",
            "factual_passes",
            "cf_crashes",
            "cf_passes",
            "rule_gen_crashes",
            "rule_gen_passes",
            "reference_crashes",
            "reference_passes",
            "collision_agreement",
            "real_interactions",
            "model_interactions",
        )
    }
    t_start = time.time()
    validation_env = {
        "PIXEL_EVAL_BASE_CACTUS_SPAWN_PROB": str(envs.BASE_CACTUS_SPAWN_PROB),
        "PIXEL_EVAL_MAX_CACTUS_SPAWN_PROB": str(envs.MAX_CACTUS_SPAWN_PROB),
        "PIXEL_EVAL_BIRD_SPAWN_PROB": str(envs.BIRD_SPAWN_PROB),
        "PIXEL_EVAL_BIRD_MIN_SPEED": str(envs.BIRD_MIN_SPEED),
        "PIXEL_EVAL_BIRD_Y_POSITIONS": ",".join(map(str, envs.BIRD_Y_POSITIONS)),
    }

    for update_i in range(args.updates):
        started = time.time()
        dump = bool(args.dump_every and update_i % args.dump_every == 0)
        episodes, stats, frames = collect_world_model_groups(
            world,
            reward_model,
            policy,
            rng,
            args.num_groups,
            args.group_size,
            args.decisions,
            args.method,
            args.collision_penalty,
            args.action_repeat,
            root_ttc_bins,
            root_speed_bins,
            keep_frames=dump,
        )
        collect_seconds = time.time() - started
        if dump:
            rollouts = os.path.join(out_dir, "rollouts")
            os.makedirs(rollouts, exist_ok=True)
            save_rollout_grid(
                os.path.join(rollouts, f"update_{update_i:05d}.png"), frames
            )

        advantages = leave_one_out_advantages(episodes, args.gamma, args.group_size)
        optimizer.zero_grad()
        loss, _, entropy_means = leave_one_out_loss(
            episodes, advantages, args.entropy_coef, DEVICE
        )
        loss.backward()
        grad_norm = torch.sqrt(
            sum(
                (parameter.grad.detach() ** 2).sum()
                for parameter in policy.parameters()
                if parameter.grad is not None
            )
        ).item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()

        total_rewards = np.asarray(
            [episode.total_reward for episode in episodes]
        )
        values = {
            "reward": total_rewards.mean(),
            "entropy": np.mean(entropy_means),
            "gradnorm": grad_norm,
            "group_spread": np.mean(
                total_rewards.reshape(args.num_groups, args.group_size).std(1)
            ),
            "factual_crashes": np.mean(stats["factual_crashes"]),
            "factual_passes": np.mean(stats["factual_passes"]),
            "cf_crashes": np.mean(stats["cf_crashes"]),
            "cf_passes": np.mean(stats["cf_passes"]),
            "rule_gen_crashes": np.mean(stats["rule_gen_crashes"]),
            "rule_gen_passes": np.mean(stats["rule_gen_passes"]),
            "reference_crashes": np.mean(stats["reference_crashes"]),
            "reference_passes": np.mean(stats["reference_passes"]),
            "collision_agreement": np.mean(stats["collision_agreement"]),
            "real_interactions": stats["real_interactions"],
            "model_interactions": stats["model_interactions"],
        }
        for key, value in values.items():
            hist[key].append(float(value))

        if update_i % 5 == 0 or update_i == args.updates - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (update_i + 1) * (args.updates - update_i - 1) / 3600
            print(
                f"[{args.method}] upd {update_i:5d}  "
                f"avg_reward={values['reward']:6.2f}  "
                f"factual C/P={values['factual_crashes']:.2f}/{values['factual_passes']:.2f}  "
                f"pred C/P={values['cf_crashes']:.2f}/{values['cf_passes']:.2f}  "
                f"ref C/P={values['reference_crashes']:.2f}/{values['reference_passes']:.2f}  "
                f"rule C/P={values['rule_gen_crashes']:.2f}/{values['rule_gen_passes']:.2f}  "
                f"agree(gen)={100 * values['collision_agreement']:.1f}%  "
                f"entropy={values['entropy']:.4f} |g|={grad_norm:7.2f}  "
                f"collect={collect_seconds:.1f}s ETA={eta:.1f}h",
                flush=True,
            )

        checkpoint = update_i + 1
        if checkpoint % args.save_every == 0 or checkpoint == args.updates:
            torch.save(
                policy.state_dict(), os.path.join(out_dir, f"policy-{checkpoint}.pth")
            )
            for key, history_values in hist.items():
                np.save(
                    os.path.join(out_dir, f"{key}_history.npy"),
                    np.asarray(history_values),
                )
            if args.auto_validate:
                auto_validate_checkpoint(
                    args.results_dir,
                    output_root,
                    validation_log_dir,
                    checkpoint,
                    args.action_repeat,
                    args.auto_validate_episodes,
                    args.auto_validate_episode_seed,
                    validation_env,
                )

    elapsed = time.time() - t_start
    print(
        f"done: {args.updates} updates in {elapsed / 3600:.2f}h "
        f"({elapsed / args.updates:.1f}s/update)"
    )


def parser():
    p = argparse.ArgumentParser(
        description="Train RLOO + WM or RLB + CWM with one simulator factual per group"
    )
    p.add_argument("--method", choices=("rlb_cwm", "rloo_wm"), required=True)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--num-groups", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--decisions", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coef", type=float, default=0.04)
    p.add_argument("--max-grad-norm", type=float, default=20.0)
    p.add_argument("--collision-penalty", type=float, default=-1.0)
    p.add_argument("--action-repeat", type=int, default=4)
    p.add_argument("--root-ttc-bins", default="8-12")
    p.add_argument("--root-speed-bins", default="21-26,27-31,32-37,38-43")
    p.add_argument("--bird-spawn-prob", type=float, default=0.6)
    p.add_argument("--bird-min-speed", type=float, default=20.0)
    p.add_argument("--mid-bird-weight", type=int, default=2)
    p.add_argument("--model-folder", required=True)
    p.add_argument("--reward-model", required=True)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--vae-chunk", type=int, default=32)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--dump-every", type=int, default=200)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument(
        "--output-root",
        default=None,
        help="checkpoint root (default: runs/policies/checkpoints)",
    )
    p.add_argument("--validation-log-dir", default=None)
    p.add_argument("--auto-validate", action="store_true")
    p.add_argument("--auto-validate-episodes", type=int, default=30)
    p.add_argument("--auto-validate-episode-seed", type=int, default=2200000000)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
