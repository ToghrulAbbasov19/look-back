"""Generate scripted or random simulator episodes for reward-model training."""

import argparse
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from chrome_dino_game import envs
from oracle import DINO_X, oracle_action


class CountingWrapper(envs.Wrapper):
    """Track collisions and passed obstacles for each decision."""

    def step(self, action):
        total, crashes, passes, terminated = 0.0, 0, 0, False
        for _ in range(self.k):
            obs, reward, term, *_ = self.env.step(action)
            self.frames.append(obs)
            self.stack.append(self._transform(obs))
            total += float(reward)
            crashes += int(reward < 0)
            passes += int(reward > 0)
            if term:
                terminated = True
                break
        self.last_counts = (crashes, passes)
        return self._get_obs(), total, terminated, False, {}


def save_episode(
    out_dir,
    frames,
    actions,
    rewards,
    episode_idx,
    gif_every=0,
    crashes=None,
    passes=None,
    save_every=1,
    deviated=None,
):
    episode_dir = Path(out_dir) / f"episode_{episode_idx:06d}"
    frame_dir = episode_dir / "frames"

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        if i % save_every == 0:
            Image.fromarray(frame).save(frame_dir / f"{i:06d}.png")

    np.save(episode_dir / "actions.npy", np.array(actions, dtype=np.int64))
    np.save(episode_dir / "rewards.npy", np.array(rewards, dtype=np.float32))
    if crashes is not None:
        np.save(episode_dir / "crashes.npy", np.array(crashes, dtype=np.int64))
        np.save(episode_dir / "passes.npy", np.array(passes, dtype=np.int64))
    if deviated is not None and any(deviated):
        np.save(episode_dir / "deviated.npy", np.array(deviated, dtype=np.int64))

    if gif_every and episode_idx % gif_every == 0:
        import imageio

        imageio.mimsave(episode_dir / "preview.gif", frames[::4], fps=10)


def next_obstacle(raw):
    pending = [o for o in raw._obstacles if o.needs_collision_check]
    return min(pending, key=lambda o: o.rect.x) if pending else None


def frames_to_next_obstacle(raw) -> float:
    nxt = next_obstacle(raw)
    return float("inf") if nxt is None else (nxt.rect.x - DINO_X) / max(raw._speed, 1)


def collect_episode(
    env,
    raw,
    policy: str,
    epsilon: float,
    rng,
    max_decisions: int,
    jump_lo: float = 0.0,
    jump_hi: float = 8.0,
    filler_prob: float = 0.0,
    safe_ttc: float = 24.0,
    k: int = 4,
    deviate_prob: float = 0.0,
    deviate_ttc: float = 16.0,
):
    env.reset()
    actions, rewards, crashes, passes, deviated = [], [], [], [], []
    decided = []

    for _ in range(max_decisions):
        deviating = False
        nxt = next_obstacle(raw) if deviate_prob > 0 else None
        if (
            nxt is not None
            and (nxt.rect.x - DINO_X) / max(raw._speed, 1) <= deviate_ttc
        ):
            flag = next((f for o, f in decided if o is nxt), None)
            if flag is None:
                flag = bool(rng.random() < deviate_prob)
                decided.append((nxt, flag))
            deviating = flag

        if policy == "random" or deviating or rng.random() < epsilon:
            a = int(rng.integers(0, 3))
        else:
            a = int(oracle_action(raw, jump_ttc=jump_hi, jump_lo=jump_lo))
            airborne = (
                raw._agent.state == envs.DinoState.JUMP and raw._agent._jump_timer >= k
            )
            if (
                a == envs.Action.STAND
                and rng.random() < filler_prob
                and (airborne or frames_to_next_obstacle(raw) > safe_ttc)
            ):
                a = int(rng.choice([envs.Action.JUMP, envs.Action.DUCK]))

        _, r, term, trunc, _ = env.step(envs.Action(a))
        actions.append(a)
        rewards.append(float(r))
        crashes.append(env.last_counts[0])
        passes.append(env.last_counts[1])
        deviated.append(int(deviating))
        if term or trunc:
            break

    return env.frames, actions, rewards, crashes, passes, deviated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["oracle", "random"], required=True)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Probability of a uniform-random action, applied on top of "
        "the oracle. A little (0.05-0.1) gives the world model "
        "recovery states it would never see from perfect play; "
        "--policy random ignores this and is fully random.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=1500,
        help="Game-frame cap per episode. Raises the 300 registered on "
        "Env-v0, which the oracle would otherwise hit constantly.",
    )
    p.add_argument("--k", type=int, default=4, help="action repeat / frames per stack")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gif-every",
        type=int,
        default=0,
        help="Write a preview GIF every N episodes (0 = never).",
    )
    p.add_argument(
        "--bird-spawn-prob",
        type=float,
        default=None,
        help="Override envs.BIRD_SPAWN_PROB (game default 0.1).",
    )
    p.add_argument(
        "--bird-min-speed",
        type=int,
        default=None,
        help="Override envs.BIRD_MIN_SPEED (game default 28; speed starts at 20).",
    )
    p.add_argument(
        "--mid-bird-weight",
        type=int,
        default=1,
        help="Relative weight of MID birds vs LOW/HIGH (1 = uniform, 2 = half MID).",
    )
    p.add_argument(
        "--filler-prob",
        type=float,
        default=0.0,
        help="When the oracle would Stand and it is safe, take a random Jump/Duck "
        "with this probability (adds action variety not tied to obstacles).",
    )
    p.add_argument(
        "--safe-ttc",
        type=float,
        default=24.0,
        help="Filler actions on the ground only when the next obstacle is more than "
        "this many frames away (a jump lasts 13 frames).",
    )
    p.add_argument(
        "--game-mode",
        choices=["normal", "train"],
        default="normal",
        help="normal: an episode ends at the first collision. train: a collision costs -1 and "
        "play continues until --max-frames, so the videos show what follows a crash.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save only every N-th game frame (4 = decision frames only; file names keep "
        "game-frame indices, so decision t is the t-th file).",
    )
    p.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Number the first episode this; use it to continue an interrupted run in the same "
        "--out without overwriting (and pass a different --seed).",
    )
    p.add_argument(
        "--deviate-prob",
        type=float,
        default=0.0,
        help="Oracle only: probability that the policy deviates at an obstacle, taking uniformly "
        "random actions from when it is --deviate-ttc frames away until it has passed.",
    )
    p.add_argument(
        "--deviate-ttc",
        type=float,
        default=16.0,
        help="Frames to contact at which the deviation decision is made (a jump lasts 13 frames; "
        "the oracle jumps at 5-9).",
    )
    a = p.parse_args()

    if a.bird_spawn_prob is not None:
        envs.BIRD_SPAWN_PROB = a.bird_spawn_prob
    if a.bird_min_speed is not None:
        envs.BIRD_MIN_SPEED = a.bird_min_speed
    low, mid, high = envs.BIRD_Y_POSITIONS[:3]
    envs.BIRD_Y_POSITIONS = (low, *[mid] * a.mid_bird_weight, high)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    env = gym.make(
        "Env-v0",
        render_mode="rgb_array",
        max_episode_steps=a.max_frames,
        game_mode=a.game_mode,
        train_frame_limit=a.max_frames,
    )
    raw = env.unwrapped
    env = CountingWrapper(env, k=a.k)

    max_decisions = a.max_frames // a.k
    jump_lo, jump_hi = (5.0, 9.0) if a.k > 1 else (0.0, 8.0)
    print(f"oracle jump window: {jump_lo}-{jump_hi} frames to contact")
    print(
        f"policy={a.policy} epsilon={a.epsilon} filler_prob={a.filler_prob} safe_ttc={a.safe_ttc} "
        f"deviate_prob={a.deviate_prob} deviate_ttc={a.deviate_ttc} k={a.k} "
        f"max {a.max_frames} frames = {max_decisions} decisions/episode, "
        f"game mode {a.game_mode}, saving every {a.save_every} frame(s)"
    )
    print(
        f"env: JUMP_VEL={envs.JUMP_VEL} MAX_SPEED={envs.MAX_SPEED} "
        f"BIRD_SPAWN_PROB={envs.BIRD_SPAWN_PROB} BIRD_MIN_SPEED={envs.BIRD_MIN_SPEED} "
        f"BIRD_Y_POSITIONS={envs.BIRD_Y_POSITIONS}"
    )

    lens, rets, n_crash, n_pass, n_dev = [], [], [], [], []
    for ep in range(a.episodes):
        frames, actions, rewards, crashes, passes, deviated = collect_episode(
            env,
            raw,
            a.policy,
            a.epsilon,
            rng,
            max_decisions,
            jump_lo,
            jump_hi,
            filler_prob=a.filler_prob,
            safe_ttc=a.safe_ttc,
            k=a.k,
            deviate_prob=a.deviate_prob,
            deviate_ttc=a.deviate_ttc,
        )
        save_episode(
            a.out,
            frames,
            actions,
            rewards,
            a.start_episode + ep,
            a.gif_every,
            crashes=crashes,
            passes=passes,
            save_every=a.save_every,
            deviated=deviated,
        )
        lens.append(len(frames))
        rets.append(sum(rewards))
        n_crash.append(sum(crashes))
        n_pass.append(sum(passes))
        n_dev.append(sum(deviated))
        if (ep + 1) % 25 == 0 or ep + 1 == a.episodes:
            print(
                f"  {ep + 1}/{a.episodes}  mean frames {np.mean(lens[-25:]):7.1f}  "
                f"mean return {np.mean(rets[-25:]):6.2f}  crashes {np.mean(n_crash[-25:]):5.1f}  "
                f"passes {np.mean(n_pass[-25:]):5.1f}  deviated decisions {np.mean(n_dev[-25:]):5.1f}",
                flush=True,
            )

    env.close()
    print(f"\ndone: {a.episodes} episodes -> {a.out}")
    print(
        f"  mean frames/episode {np.mean(lens):.1f}   mean return {np.mean(rets):.2f}"
    )


if __name__ == "__main__":
    main()
