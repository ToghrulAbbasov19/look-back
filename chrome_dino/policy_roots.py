"""Construct TTC-aligned simulator roots."""

from collections import deque

import numpy as np

from action_repeat import RepeatEnv
from common import envs
from policy_core import K

DINO_FRONT_X = 138


def advance_to_obstacle(
    e,
    seed,
    history: int = 0,
    target_ttc: float = None,
    target_speed=None,
):
    """Advance with STAND to a standing root at the requested TTC and speed."""
    target_ttc = float(envs.JUMP_DURATION if target_ttc is None else target_ttc)
    assert target_ttc > 0
    raw = getattr(e, "unwrapped", e)
    base = e.env if isinstance(e, RepeatEnv) else e
    saved_render = raw._render_frame
    raw._render_frame = lambda: None
    recent = deque(maxlen=max(history, 1))
    defer_history_render = target_speed is not None and history > 0
    found = False
    try:
        attempts = 32 if target_speed is not None else 1
        for attempt in range(attempts):
            attempt_seed = int((int(seed) + attempt * 1_000_003) % (2**31 - 1))
            e.reset(seed=attempt_seed)
            recent.clear()
            for _ in range(800):
                if target_speed is not None and raw._speed > target_speed[1]:
                    break
                near = False
                pending = [o for o in raw._obstacles if o.needs_collision_check]
                if pending:
                    gap = min(pending, key=lambda o: o.rect.x).rect.x - DINO_FRONT_X
                    ttc = gap / max(raw._speed, 1)
                    standing = raw._agent.state == envs.DinoState.STAND
                    speed_ok = (
                        target_speed is None
                        or target_speed[0] <= raw._speed <= target_speed[1]
                    )
                    if target_ttc - 1 < ttc <= target_ttc and standing and speed_ok:
                        found = True
                        break
                    near = history > 0 and ttc <= target_ttc + history + 1
                _, _, term, *_ = base.step(envs.Action.STAND)
                if term:
                    e.reset(seed=attempt_seed)
                    recent.clear()
                elif near:
                    if defer_history_render:
                        recent.append(
                            (
                                raw.get_state(),
                                tuple(raw._assets.birds),
                                tuple(raw._assets.dino_runs),
                                tuple(raw._assets.dino_ducks),
                            )
                        )
                        track = raw._track
                        if track._track_offset_x + track._track_w < envs.WINDOW_SIZE[0]:
                            start_x = track._track_offset_x + track._track_w - 10
                            if start_x <= 0:
                                track._track_offset_x = start_x
                    else:
                        recent.append(saved_render())
            if found:
                break
    finally:
        raw._render_frame = saved_render
    if not found:
        raise RuntimeError(
            f"could not construct a valid root after {attempts} attempt(s): "
            f"TTC target={target_ttc}, speed={target_speed}"
        )
    if defer_history_render:
        root_state = raw.get_state()
        root_birds = tuple(raw._assets.birds)
        root_runs = tuple(raw._assets.dino_runs)
        root_ducks = tuple(raw._assets.dino_ducks)
        rendered_recent = []
        try:
            for state, birds, runs, ducks in recent:
                raw.set_state(state)
                raw._assets.birds[:] = birds
                raw._assets.dino_runs[:] = runs
                raw._assets.dino_ducks[:] = ducks
                rendered_recent.append(raw._render_frame())
        finally:
            raw.set_state(root_state)
            raw._assets.birds[:] = root_birds
            raw._assets.dino_runs[:] = root_runs
            raw._assets.dino_ducks[:] = root_ducks
        recent = rendered_recent
    return raw._render_frame(), list(recent)


def seed_stack(final, recent, action_repeat: int):
    """Build the decision-spaced frame stack ending at the root."""
    frames = list(recent) + [final]
    return [
        frames[max(len(frames) - 1 - j * action_repeat, 0)]
        for j in range(K - 1, -1, -1)
    ]


def parse_root_ttc_bins(spec: str) -> list[tuple[int, int]]:
    """Parse TTC intervals such as `8-12`, interpreted as (8, 12]."""
    if not spec:
        return []
    bins = []
    for item in spec.split(","):
        lo_s, sep, hi_s = item.strip().partition("-")
        if not sep:
            raise ValueError("--root-ttc-bins must look like 8-12,16-24,28-36")
        lo, hi = int(lo_s), int(hi_s)
        if lo <= 0 or hi < lo:
            raise ValueError(f"invalid TTC bin {item!r}")
        bins.append((lo, hi))
    return bins


def parse_root_speed_bins(spec: str) -> list[tuple[int, int]]:
    """Parse integer simulator-speed bins such as `21-26,27-31,32-37,38-43`."""
    if not spec:
        return []
    bins = []
    for item in spec.split(","):
        lo_s, sep, hi_s = item.strip().partition("-")
        if not sep:
            raise ValueError(
                "--root-speed-bins must look like 21-26,27-31,32-37,38-43"
            )
        lo, hi = int(lo_s), int(hi_s)
        if lo <= 0 or hi < lo:
            raise ValueError(f"invalid speed bin {item!r}")
        bins.append((lo, hi))
    return bins


def _sample_root_ttc(rng, ttc_bins, bin_index=None, lower_exclusive=False):
    if not ttc_bins:
        return float(envs.JUMP_DURATION)
    if bin_index is None:
        bin_index = int(rng.integers(len(ttc_bins)))
    lo, hi = ttc_bins[bin_index]
    lower = lo + int(lower_exclusive)
    if lower > hi:
        raise ValueError("a lower-exclusive TTC bin must contain an integer")
    return float(rng.integers(lower, hi + 1))


def _balanced_root_ttcs(rng, count, ttc_bins, lower_exclusive=False):
    """Sample exact TTCs with every requested bin represented as evenly as possible."""
    if not ttc_bins:
        return [_sample_root_ttc(rng, ()) for _ in range(count)]
    indices = np.arange(count) % len(ttc_bins)
    rng.shuffle(indices)
    return [_sample_root_ttc(rng, ttc_bins, int(i), lower_exclusive) for i in indices]


def _balanced_root_speeds(rng, count, speed_bins):
    """Assign requested root-speed bins equally within every update."""
    if not speed_bins:
        return [None] * count
    indices = np.arange(count) % len(speed_bins)
    rng.shuffle(indices)
    return [speed_bins[int(i)] for i in indices]


def build_root_seeds(
    rng,
    n,
    gsz,
    grouped,
    root_ttc_bins=(),
    independent_future_noise=False,
    root_speed_bins=(),
):
    """Seeds, TTCs, and speeds for one batch of roots.

    grouped=True: each group of gsz rollouts starts from the same root. With
    independent_future_noise (RLOO) every member then gets its own obstacle RNG;
    without it (RLB) the members also share the future obstacles.
    """
    seeds, ttcs, noise_seeds, speed_targets = [], [], [], []
    n_groups = n // gsz
    n_starts = n_groups if grouped else n
    # Unused, but it advances rng; kept so seeded runs reproduce the reported roots.
    root_slots = np.ones(n_starts, dtype=bool)
    rng.shuffle(root_slots)
    start_ttcs = _balanced_root_ttcs(
        rng, n_starts, root_ttc_bins, lower_exclusive=True
    )
    start_speeds = _balanced_root_speeds(rng, n_starts, root_speed_bins)

    for group_i in range(n_groups):
        if grouped:
            gs = int(rng.integers(0, 2**31 - 1))
            seeds += [gs] * gsz
            ttcs += [start_ttcs[group_i]] * gsz
            speed_targets += [start_speeds[group_i]] * gsz
            noise_seeds += (
                [int(rng.integers(0, 2**31 - 1)) for _ in range(gsz)]
                if independent_future_noise
                else [None] * gsz
            )
        else:
            offset = group_i * gsz
            seeds += [int(rng.integers(0, 2**31 - 1)) for _ in range(gsz)]
            ttcs += start_ttcs[offset : offset + gsz]
            speed_targets += start_speeds[offset : offset + gsz]
            noise_seeds += [None] * gsz
    return seeds, ttcs, noise_seeds, speed_targets


def reseed_env_noise(e, noise_seed):
    """Set the RNG used for future obstacle generation."""
    if noise_seed is None:
        return
    raw = getattr(e, "unwrapped", e)
    raw.np_random.bit_generator.state = np.random.default_rng(
        int(noise_seed)
    ).bit_generator.state
