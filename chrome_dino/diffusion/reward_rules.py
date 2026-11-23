"""Infer Dino rewards from frames without stepping the simulator.

Obstacles are tracked visually while Dino pose and game speed follow the action history.
"""

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

FULL_W, FULL_H = 1024, 512
DINO_X = 50
RUN_SIZE, DUCK_SIZE = (88, 94), (118, 60)
JUMP_VEL, JUMP_DURATION = 38, 12
COLLISION_THRESHOLD = 20
FRAMES_PER_DECISION = 4
STAND, JUMP, DUCK = 0, 1, 2
MIN_SPEED, MAX_SPEED, SPEEDUP_EVERY = 20, 43, 20
BIRD_HEIGHT = 90

DARK_THRESHOLD = 210  # Preserve thin, downsampled cactus trunks.
SCORE_ROWS = 45
DINO_ZONE_RIGHT = 190  # Ignore detections overlapping the Dino.
MIN_OBSTACLE_HEIGHT = 40  # Reject track texture.
FIT_WINDOW = 80


def detect_obstacles(frame) -> list[tuple[float, float, float, float, bool]]:
    """Detect obstacles in simulator coordinates."""
    gray = np.asarray(frame, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    h, w = gray.shape
    sx, sy = FULL_W / w, FULL_H / h
    dark = gray < DARK_THRESHOLD

    lo = int(0.85 * h)
    row_dark = dark[lo:].mean(axis=1)
    ground = lo + int(
        np.argmax(row_dark > 0.5) if (row_dark > 0.5).any() else row_dark.argmax()
    )
    y0, x0 = int(SCORE_ROWS / sy), int(DINO_ZONE_RIGHT / sx)
    region = dark[y0:ground, x0:]
    labels, _ = ndimage.label(
        ndimage.binary_dilation(region, iterations=max(1, round(3 / sx)))
    )
    labels[~region] = 0

    boxes = []
    for sl in ndimage.find_objects(labels):
        top, bottom = (sl[0].start + y0) * sy, (sl[0].stop + y0) * sy
        left, right = (sl[1].start + x0) * sx, (sl[1].stop + x0) * sx
        if bottom - top < MIN_OBSTACLE_HEIGHT or sl[1].start == 0:
            continue
        clipped = sl[1].stop == region.shape[1]
        if sl[0].stop + y0 >= ground - 1:
            boxes.append((left - 1, right + 2, top - 4, FULL_H, clipped))
        else:
            boxes.append(
                (left - 5.5, right + 3, top - 6.5, top - 6.5 + BIRD_HEIGHT, clipped)
            )
    return boxes


def _speed_candidates(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute displacement for each initial speed and phase."""
    starts = np.arange(MIN_SPEED, MAX_SPEED + 1)
    phases = np.arange(0, SPEEDUP_EVERY, FRAMES_PER_DECISION)
    f = np.arange(1, n_frames + 1)
    speed = np.minimum(
        MAX_SPEED,
        starts[:, None, None]
        + (f[None, None, :] + phases[None, :, None]) // SPEEDUP_EVERY,
    )
    cum = np.concatenate(
        [np.zeros(speed.shape[:2] + (1,)), np.cumsum(speed, axis=2)], axis=2
    )
    return cum.reshape(-1, n_frames + 1), speed.reshape(-1, n_frames)


@dataclass
class Track:

    anchor: float
    top: float
    bottom: float
    last_seen: int
    sightings: list = field(default_factory=list)
    widths: list = field(default_factory=list)
    done: bool = False

    @property
    def seen(self) -> int:
        return len(self.sightings)

    @property
    def width(self) -> float:
        return float(np.median(self.widths)) if self.widths else 0.0


class DinoRewardRule:
    def __init__(self, min_seen: int = 2, max_frames: int = 4096):
        self.min_seen = min_seen
        self.cum_candidates, self.speed_candidates = _speed_candidates(max_frames)
        self.reset()

    def reset(self, context_frames=(), context_actions=()):
        self.tracks: list[Track] = []
        self.model = None
        self.frame = 0
        self.state, self.timer = STAND, 0
        self.decision = 0
        for frame, action in zip(context_frames, context_actions):
            self.step(frame, action)

    @property
    def speed(self) -> float:
        if self.model is None:
            return float(MIN_SPEED)
        return float(
            self.speed_candidates[
                self.model, min(self.frame, self.speed_candidates.shape[1] - 1)
            ]
        )

    @property
    def speed_known(self) -> bool:
        return self.model is not None

    def _cum(self, frame: int) -> float:
        if self.model is None:
            return float(MIN_SPEED * frame)
        return float(self.cum_candidates[self.model, frame])

    def left(self, t: Track) -> float:
        return t.anchor - self._cum(self.frame)

    def _fit_speed(self):
        tracks = [
            t
            for t in self.tracks
            if len([s for s in t.sightings if s[0] >= self.frame - FIT_WINDOW]) >= 2
        ]
        if not tracks:
            return
        cost = np.zeros(len(self.cum_candidates))
        for t in tracks:
            f, left = np.array(
                [s for s in t.sightings if s[0] >= self.frame - FIT_WINDOW]
            ).T
            anchors = left[None, :] + self.cum_candidates[:, f.astype(int)]
            cost += np.abs(anchors - np.median(anchors, axis=1, keepdims=True)).sum(
                axis=1
            )
        self.model = int(np.argmin(cost))

    def _refresh_anchors(self):
        for t in self.tracks:
            if t.sightings:
                t.anchor = float(
                    np.median([left + self._cum(f) for f, left in t.sightings])
                )

    def _observe(self, frame):
        dets = detect_obstacles(frame)
        tol = 40.0 if self.speed_known else 120.0
        pairs = sorted(
            (abs(self.left(t) - d[0]), ti, di)
            for ti, t in enumerate(self.tracks)
            for di, d in enumerate(dets)
            if abs(self.left(t) - d[0]) < tol and abs(t.top - d[2]) < 40
        )
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._add_sighting(self.tracks[ti], dets[di])
        for di, d in enumerate(dets):
            if di not in used_d:
                t = Track(anchor=0.0, top=d[2], bottom=d[3], last_seen=self.decision)
                self._add_sighting(t, d)
                self.tracks.append(t)
        self._fit_speed()
        self._refresh_anchors()
        self.tracks = [
            t
            for t in self.tracks
            if not (
                t.seen < self.min_seen
                and self.decision - t.last_seen >= 3
                and self.left(t) > 300
            )
        ]

    def _add_sighting(self, t: Track, det):
        left, right, top, bottom, clipped = det
        t.sightings.append((self.frame, left))
        t.last_seen, t.top, t.bottom = self.decision, top, bottom
        if not clipped:
            t.widths.append(right - left)

    def _dino_step(self, action: int):
        if self.state == JUMP:
            self.timer -= 1
            if self.timer < 0:
                self.state = STAND
        if self.state != JUMP:
            if action == JUMP:
                self.state, self.timer = JUMP, JUMP_DURATION
            else:
                self.state = DUCK if action == DUCK else STAND

    def _dino_box(self):
        w, h = DUCK_SIZE if self.state == DUCK else RUN_SIZE
        offset = 0
        if self.state == JUMP:
            t = JUMP_DURATION - self.timer
            offset = int(JUMP_VEL * t - 0.5 * (JUMP_VEL / (JUMP_DURATION / 2)) * t * t)
        top = FULL_H - offset - h
        c = COLLISION_THRESHOLD
        return DINO_X + c, top + c, DINO_X + w - c, top + h - c

    def step(self, frame, action: int) -> tuple[float, bool]:
        """Compute reward over one action repeat."""
        self._observe(frame)
        reward, done = 0.0, False
        for _ in range(FRAMES_PER_DECISION):
            self.frame += 1
            self._dino_step(int(action))
            left, top, right, bottom = self._dino_box()
            for t in self.tracks:
                if t.done or t.seen < self.min_seen or not t.widths:
                    continue
                o_left = self.left(t)
                if (
                    o_left < right
                    and o_left + t.width > left
                    and t.top < bottom
                    and t.bottom > top
                ):
                    t.done, done = True, True
                    reward -= 1.0
                elif DINO_X > o_left + t.width:
                    t.done = True
                    reward += 1.0
            if done:
                break
        self.tracks = [t for t in self.tracks if self.left(t) + max(t.width, 1.0) > 0]
        self.decision += 1
        return reward, done


def validate_episode(
    ep_dir: Path, size, max_decisions: int, rule: DinoRewardRule
) -> dict:
    actions = np.load(ep_dir / "actions.npy")
    rewards = np.load(ep_dir / "rewards.npy")
    frame_paths = sorted((ep_dir / "frames").glob("*.png"))
    n = len(actions)
    crashed = n < max_decisions
    true_passes = rewards + np.where(np.arange(n) == n - 1, float(crashed), 0.0)

    rule.reset()
    pred_passes, pred_crash = np.zeros(n), None
    for t in range(n):
        frame = (
            Image.open(frame_paths[t * FRAMES_PER_DECISION])
            .convert("L")
            .resize(size, Image.BILINEAR)
        )
        r, done = rule.step(np.asarray(frame), int(actions[t]))
        pred_passes[t] = r + (1.0 if done else 0.0)
        if done and pred_crash is None:
            pred_crash = t

    horizon = min(n, pred_crash + 1 if pred_crash is not None else n)
    return {
        "episode": ep_dir.name,
        "decisions": n,
        "crashed": crashed,
        "true_crash": n - 1 if crashed else None,
        "pred_crash": pred_crash,
        "true_passes": float(true_passes[:horizon].sum()),
        "pred_passes": float(pred_passes[:horizon].sum()),
        "pass_step_match": float(
            (true_passes[:horizon] == pred_passes[:horizon]).mean()
        ),
    }


def summarize(name: str, results: list[dict]):
    crashed = [r for r in results if r["crashed"]]
    survived = [r for r in results if not r["crashed"]]
    exact = sum(r["pred_crash"] == r["true_crash"] for r in crashed)
    within1 = sum(
        r["pred_crash"] is not None and abs(r["pred_crash"] - r["true_crash"]) <= 1
        for r in crashed
    )
    early = sum(
        r["pred_crash"] is not None and r["pred_crash"] < r["true_crash"] - 1
        for r in crashed
    )
    missed = sum(
        r["pred_crash"] is None or r["pred_crash"] > r["true_crash"] + 1
        for r in crashed
    )
    false_crash = sum(r["pred_crash"] is not None for r in survived)
    tp, pp = (
        sum(r["true_passes"] for r in results),
        sum(r["pred_passes"] for r in results),
    )
    print(
        f"\n== {name}: {len(results)} episodes ({len(crashed)} crashed, {len(survived)} survived)"
    )
    if crashed:
        print(
            f"  crash step: exact {exact}/{len(crashed)} ({exact / len(crashed):.0%}), within +-1 {within1}/{len(crashed)} "
            f"({within1 / len(crashed):.0%}), too early {early}, missed/late {missed}"
        )
    if survived:
        print(f"  false crash in surviving episodes: {false_crash}/{len(survived)}")
    print(
        f"  passes (+1) up to the predicted end: true {tp:.0f}, predicted {pp:.0f}; "
        f"per-decision pass agreement {np.mean([r['pass_step_match'] for r in results]):.1%}"
    )


def main(args):
    rng = random.Random(args.seed)
    size = (args.width, args.height)
    for d in args.episodes_dir:
        episodes = sorted(Path(d).glob("episode_*"))
        episodes = sorted(rng.sample(episodes, min(args.num_episodes, len(episodes))))
        rule = DinoRewardRule(min_seen=args.min_seen)
        results = [
            validate_episode(ep, size, args.max_decisions, rule) for ep in episodes
        ]
        summarize(d, results)
        if args.show_errors:
            for r in results:
                if (r["crashed"] and r["pred_crash"] != r["true_crash"]) or (
                    not r["crashed"] and r["pred_crash"] is not None
                ):
                    print("   ", r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_dir", nargs="+", required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--max_decisions", type=int, default=375)
    parser.add_argument("--min_seen", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show_errors", action="store_true")
    main(parser.parse_args())
