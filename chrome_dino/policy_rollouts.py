"""Collect simulator policy rollouts locally or in worker processes."""

import multiprocessing as mp
from collections import deque

import numpy as np
import torch

from action_repeat import RepeatEnv
from common import Episode, envs
from policy_core import K, stack_to_tensor, to_gray
from policy_roots import (
    advance_to_obstacle,
    build_root_seeds,
    reseed_env_noise,
    seed_stack,
)


def collect_group(
    gym_envs,
    policy,
    grouped: bool,
    collision_penalty: float,
    max_steps: int,
    rng: np.random.Generator,
    group_size: int,
    root_ttc_bins=(),
    independent_future_noise: bool = False,
    root_speed_bins=(),
):
    """Collect one batch with TTC-aligned standing roots."""
    n = len(gym_envs)
    episodes = [Episode() for _ in range(n)]

    seeds, ttcs, noise_seeds, speed_targets = build_root_seeds(
        rng,
        n,
        group_size,
        grouped,
        root_ttc_bins,
        independent_future_noise,
        root_speed_bins,
    )

    stacks = []
    for i, gym_env in enumerate(gym_envs):
        repeat = gym_env.k if isinstance(gym_env, RepeatEnv) else 1
        history = (K - 1) * repeat
        recent = deque(maxlen=max(history, 1))
        obs, hist = advance_to_obstacle(
            gym_env,
            seeds[i],
            history=history,
            target_ttc=ttcs[i],
            target_speed=speed_targets[i],
        )
        recent.extend(hist)
        reseed_env_noise(gym_env, noise_seeds[i])
        frames = seed_stack(obs, list(recent), repeat)
        stacks.append(deque([to_gray(f) for f in frames], maxlen=K))

    active = [True] * n
    step = 0
    while any(active) and step < max_steps:
        idx = [i for i in range(n) if active[i]]
        states = torch.stack([stack_to_tensor(stacks[i]) for i in idx])

        logits = policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        acts = actions.tolist()

        for j, i in enumerate(idx):
            obs, reward, terminated, *_ = gym_envs[i].step(envs.Action(int(acts[j])))
            reward = float(reward)
            if reward < 0:
                reward *= abs(collision_penalty)
            episodes[i].add(states[j], log_probs[j], entropies[j], reward)
            stacks[i].append(to_gray(obs))
            if terminated:
                active[i] = False
        step += 1

    return episodes


def _env_worker(
    conn,
    game_mode: str,
    n_local: int,
    spawn_overrides: dict = None,
    action_repeat: int = 1,
):
    for name, val in (spawn_overrides or {}).items():
        setattr(envs, name, val)
    envs_local = [
        envs.Env(render_mode="rgb_array", game_mode=game_mode, train_frame_limit=99999)
        for _ in range(n_local)
    ]
    if action_repeat > 1:
        envs_local = [RepeatEnv(e, k=action_repeat) for e in envs_local]
    while True:
        msg = conn.recv()
        if msg is None:
            break
        if msg[0] == "reset":
            out = []
            _, seeds, history, ttcs, noise_seeds, speed_targets = msg
            for e, seed, ttc, noise_seed, speed_target in zip(
                envs_local, seeds, ttcs, noise_seeds, speed_targets
            ):
                recent = deque(maxlen=max(history, 1))
                obs, hist = advance_to_obstacle(
                    e,
                    seed,
                    history=history,
                    target_ttc=ttc,
                    target_speed=speed_target,
                )
                recent.extend(hist)
                reseed_env_noise(e, noise_seed)
                frames = (
                    seed_stack(obs, list(recent), action_repeat)
                    if history
                    else [obs] * K
                )
                out.append([to_gray(f) for f in frames])
            conn.send(out)
        else:
            out = []
            for li, a in msg[1]:
                e = envs_local[li]
                obs, r, term, *_ = e.step(envs.Action(int(a)))
                out.append((to_gray(obs), float(r), bool(term)))
            conn.send(out)


class ParallelEnvs:

    def __init__(
        self,
        n_envs: int,
        n_workers: int = 0,
        game_mode: str = "train",
        action_repeat: int = 1,
    ):
        spawn_overrides = {
            k: getattr(envs, k)
            for k in (
                "BASE_CACTUS_SPAWN_PROB",
                "MAX_CACTUS_SPAWN_PROB",
                "BIRD_SPAWN_PROB",
                "BIRD_MIN_SPEED",
                "BIRD_Y_POSITIONS",
            )
            if hasattr(envs, k)
        }
        n_workers = n_envs if not n_workers else min(n_workers, n_envs)
        self.action_repeat = action_repeat
        base, rem = divmod(n_envs, n_workers)
        self.assign, k = [], 0
        for w in range(n_workers):
            cnt = base + (1 if w < rem else 0)
            self.assign.append(list(range(k, k + cnt)))
            k += cnt
        self.owner = {
            g: (w, li) for w, gs in enumerate(self.assign) for li, g in enumerate(gs)
        }
        self.n_envs, self.n_workers = n_envs, n_workers

        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        for w in range(n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_env_worker,
                args=(
                    child,
                    game_mode,
                    len(self.assign[w]),
                    spawn_overrides,
                    action_repeat,
                ),
                daemon=True,
            )
            p.start()
            self.pipes.append(parent)
            self.procs.append(p)

    def reset(
        self,
        seeds,
        ttcs,
        noise_seeds=None,
        speed_targets=None,
    ):
        """Reset environments at TTC-aligned standing roots."""
        history = (K - 1) * self.action_repeat
        noise_seeds = [None] * self.n_envs if noise_seeds is None else noise_seeds
        speed_targets = [None] * self.n_envs if speed_targets is None else speed_targets
        for w, gs in enumerate(self.assign):
            self.pipes[w].send(
                (
                    "reset",
                    [int(seeds[g]) for g in gs],
                    history,
                    [float(ttcs[g]) for g in gs],
                    [
                        None if noise_seeds[g] is None else int(noise_seeds[g])
                        for g in gs
                    ],
                    [speed_targets[g] for g in gs],
                )
            )
        out = [None] * self.n_envs
        for w, gs in enumerate(self.assign):
            for g, frames in zip(gs, self.pipes[w].recv()):
                out[g] = frames
        return out

    def step(self, idx, actions):
        """Step only the envs in `idx`; results come back in that same order."""
        per = [[] for _ in range(self.n_workers)]
        for g, a in zip(idx, actions):
            w, li = self.owner[g]
            per[w].append((li, int(a)))
        for w in range(self.n_workers):
            if per[w]:
                self.pipes[w].send(("step", per[w]))
        got = {}
        for w in range(self.n_workers):
            if per[w]:
                for (li, _), res in zip(per[w], self.pipes[w].recv()):
                    got[self.assign[w][li]] = res
        return [got[g] for g in idx]

    def close(self):
        for c in self.pipes:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def collect_group_parallel(
    penvs,
    policy,
    grouped: bool,
    collision_penalty: float,
    max_steps: int,
    rng: np.random.Generator,
    n: int,
    group_size: int,
    root_ttc_bins=(),
    independent_future_noise: bool = False,
    root_speed_bins=(),
):
    episodes = [Episode() for _ in range(n)]

    seeds, ttcs, noise_seeds, speed_targets = build_root_seeds(
        rng,
        n,
        group_size,
        grouped,
        root_ttc_bins,
        independent_future_noise,
        root_speed_bins,
    )

    grays = penvs.reset(
        seeds,
        ttcs,
        noise_seeds,
        speed_targets,
    )
    stacks = [deque(frames, maxlen=K) for frames in grays]

    active = [True] * n
    step = 0
    while any(active) and step < max_steps:
        idx = [i for i in range(n) if active[i]]
        states = torch.stack([stack_to_tensor(stacks[i]) for i in idx])

        logits = policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        results = penvs.step(idx, actions.tolist())
        for j, i in enumerate(idx):
            g, reward, terminated = results[j]
            if reward < 0:
                reward *= abs(collision_penalty)
            episodes[i].add(states[j], log_probs[j], entropies[j], reward)
            stacks[i].append(g)
            if terminated:
                active[i] = False
        step += 1

    return episodes

