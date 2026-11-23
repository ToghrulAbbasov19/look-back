# Adapted from aome510/chrome-dino-game-rl (MIT); modified for this project.
"""Chrome Dino simulator (pygame) with train and normal game modes."""

from abc import ABC
from collections import deque
import enum
from pathlib import Path
from typing import Any

from PIL import Image
import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from gymnasium.envs.registration import register


ASSET_DIR = Path(__file__).resolve().parent / "assets"

WINDOW_SIZE = (1024, 512)
JUMP_DURATION = 12
JUMP_VEL = 38

BIRD_Y_POSITIONS = (437, 363, 330)

BIRD_HEIGHT = 90
OBSTACLE_MIN_CNT = 400
MAX_SPEED = 43
MAX_CACTUS_SPAWN_PROB = 0.7
BASE_CACTUS_SPAWN_PROB = 0.3
BIRD_SPAWN_PROB = 0.1

BIRD_MIN_SPEED = 28
RENDER_FPS = 15
COLLISION_THRESHOLD = 20
DIFFICULTY_INCREASE_FREQ = 20


class Action(int, enum.Enum):
    STAND = 0
    JUMP = 1
    DUCK = 2


class DinoState(int, enum.Enum):
    STAND = 0
    JUMP = 1
    DUCK = 2


class GameMode(str, enum.Enum):
    NORMAL = "normal"
    TRAIN = "train"


class RenderMode(str, enum.Enum):
    HUMAN = "human"
    RGB = "rgb_array"


def _scale_to_height(surf: pygame.Surface, height: int = None) -> pygame.Surface:
    """Stretch a sprite vertically while preserving its width."""
    h = BIRD_HEIGHT if height is None else height
    return pygame.transform.scale(surf, (surf.get_width(), h))


class Assets:
    def __init__(self):
        self.track = pygame.image.load(ASSET_DIR / "Track.png")

        self.dino_runs = [
            pygame.image.load(ASSET_DIR / "DinoRun1.png"),
            pygame.image.load(ASSET_DIR / "DinoRun2.png"),
        ]
        self.dino_ducks = [
            pygame.image.load(ASSET_DIR / "DinoDuck1.png"),
            pygame.image.load(ASSET_DIR / "DinoDuck2.png"),
        ]
        self.dino_jump = pygame.image.load(ASSET_DIR / "DinoJump.png")

        self.cactuses = [
            pygame.image.load(ASSET_DIR / "LargeCactus1.png"),
            pygame.image.load(ASSET_DIR / "LargeCactus2.png"),
            pygame.image.load(ASSET_DIR / "LargeCactus3.png"),
            pygame.image.load(ASSET_DIR / "SmallCactus1.png"),
            pygame.image.load(ASSET_DIR / "SmallCactus2.png"),
            pygame.image.load(ASSET_DIR / "SmallCactus3.png"),
        ]

        self.birds = [
            _scale_to_height(pygame.image.load(ASSET_DIR / "Bird1.png")),
            _scale_to_height(pygame.image.load(ASSET_DIR / "Bird2.png")),
        ]


class EnvObject(ABC):
    rect: pygame.Rect

    def __init__(self, assets: Assets, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def render(self, canvas: pygame.Surface, *args, **kwargs):
        pass


class Obstacle(EnvObject, ABC):
    needs_collision_check = True

    def collide(self, o: pygame.Rect) -> bool:
        return self.rect.colliderect(
            o.left + COLLISION_THRESHOLD,
            o.top + COLLISION_THRESHOLD,
            o.width - 2 * COLLISION_THRESHOLD,
            o.height - 2 * COLLISION_THRESHOLD,
        )

    def is_inside(self) -> bool:
        return False


class Bird(Obstacle):
    def __init__(self, assets: Assets, y: int | None = None):
        self._assets = assets.birds
        self.rect = self._assets[0].get_rect()
        self.rect.x = WINDOW_SIZE[0]
        self.rect.y = BIRD_Y_POSITIONS[1] if y is None else y

    def step(self, speed: int):
        self.rect.x -= speed
        self._assets[0], self._assets[1] = (
            self._assets[1],
            self._assets[0],
        )

    def is_inside(self) -> bool:
        return self.rect.x + self._assets[0].get_width() > 0

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._assets[0],
            self.rect,
        )


class Cactus(Obstacle):
    def __init__(self, assets: Assets, id: int):
        self._asset = assets.cactuses[id]
        self.rect = self._asset.get_rect()
        self.rect.x = WINDOW_SIZE[0]
        self.rect.y = WINDOW_SIZE[1] - self._asset.get_height() - 7

    def step(self, speed: int):
        self.rect.x -= speed

    def is_inside(self) -> bool:
        return self.rect.x + self._asset.get_width() > 0

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._asset,
            self.rect,
        )


class Dino(EnvObject):
    def __init__(self, assets: Assets):
        self._run_assets = assets.dino_runs
        self._duck_assets = assets.dino_ducks
        self._jump_asset = assets.dino_jump

        self._jump_timer = 0
        self.state = DinoState.STAND

    def step(self, action: Action):
        self._run_assets[0], self._run_assets[1] = (
            self._run_assets[1],
            self._run_assets[0],
        )
        self._duck_assets[0], self._duck_assets[1] = (
            self._duck_assets[1],
            self._duck_assets[0],
        )

        if self.state == DinoState.JUMP:
            self._jump_timer -= 1
            if self._jump_timer < 0:
                self.state = DinoState.STAND

        if self.state != DinoState.JUMP:
            match action:
                case Action.STAND:
                    self.state = DinoState.STAND
                case Action.JUMP:
                    self.state = DinoState.JUMP
                    self._jump_timer = JUMP_DURATION
                case Action.DUCK:
                    self.state = DinoState.DUCK

    def get_data(self) -> tuple[pygame.Surface, pygame.Rect]:
        match self.state:
            case DinoState.STAND:
                asset = self._run_assets[0]
                y = WINDOW_SIZE[1] - asset.get_height()
            case DinoState.JUMP:
                asset = self._jump_asset
                y = WINDOW_SIZE[1] - self._get_jump_offset() - asset.get_height()
            case DinoState.DUCK:
                asset = self._duck_assets[0]
                y = WINDOW_SIZE[1] - asset.get_height()

        rect = pygame.Rect(50, y, asset.get_width(), asset.get_height())

        return asset, rect

    def _get_jump_offset(self) -> int:
        a = -JUMP_VEL / (JUMP_DURATION / 2)
        t = JUMP_DURATION - self._jump_timer
        d = int(JUMP_VEL * t + 0.5 * a * (t**2))
        return d

    def render(self, canvas: pygame.Surface):
        asset, rect = self.get_data()
        canvas.blit(asset, rect)


class Track(EnvObject):
    def __init__(self, assets: Assets):
        self._asset = assets.track

        self._track_offset_x = 0
        self._track_w = self._asset.get_width()
        self._track_h = self._asset.get_height()

    def step(self, speed: int):
        self._track_offset_x -= speed

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._asset,
            (self._track_offset_x, WINDOW_SIZE[1] - self._track_h),
        )

        if self._track_offset_x + self._track_w < WINDOW_SIZE[0]:
            start_x = self._track_offset_x + self._track_w - 10
            canvas.blit(
                self._asset,
                (start_x, WINDOW_SIZE[1] - self._track_h),
            )

            if start_x <= 0:
                self._track_offset_x = start_x


class Env(gym.Env):
    metadata = {
        "render_fps": RENDER_FPS,
        "render_modes": [RenderMode.HUMAN, RenderMode.RGB],
    }

    def __init__(
        self,
        render_mode: RenderMode | None,
        game_mode: GameMode = GameMode.NORMAL,
        train_frame_limit=500,
    ) -> None:
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(len(list(Action)))
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8
        )

        self._game_mode = game_mode
        self._train_frame_limit = train_frame_limit

        self._window = None
        self._clock = None

        pygame.freetype.init()
        self._game_font = pygame.freetype.SysFont(
            pygame.freetype.get_default_font(), 24
        )
        if self.render_mode == RenderMode.HUMAN:
            pygame.init()
            pygame.display.init()

            self._window = pygame.display.set_mode(WINDOW_SIZE)
            self._clock = pygame.time.Clock()

        self._init_game_data()

        super().__init__()

    def _init_game_data(self):
        if not hasattr(self, "_assets"):
            self._assets = Assets()

        self._frame = 0
        self._speed = 20
        self._spawn_prob = BASE_CACTUS_SPAWN_PROB
        self._obstacle_cnt = OBSTACLE_MIN_CNT

        self._track = Track(self._assets)
        self._agent = Dino(self._assets)
        self._obstacles: list[Obstacle] = []

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        self._init_game_data()

        obs = self._render_frame()

        return obs, {}

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict]:
        terminated = False
        reward = 0.0

        self._frame += 1
        self._obstacle_cnt += self._speed
        if self._frame % DIFFICULTY_INCREASE_FREQ == 0:
            self._speed = min(MAX_SPEED, self._speed + 1)
            self._spawn_prob = min(MAX_CACTUS_SPAWN_PROB, self._spawn_prob * 1.01)

        self._track.step(self._speed)
        self._agent.step(action)
        for o in self._obstacles:
            o.step(self._speed)

        self._obstacles = [o for o in self._obstacles if o.is_inside()]

        _, agent_rect = self._agent.get_data()
        for o in self._obstacles:
            if not o.needs_collision_check:
                continue
            if o.collide(agent_rect):
                o.needs_collision_check = False
                reward -= 1.0
                if self._game_mode == GameMode.NORMAL:
                    terminated = True
            else:
                if agent_rect.left > o.rect.right:
                    o.needs_collision_check = False
                    reward += 1.0

        if self._game_mode == GameMode.TRAIN and self._frame >= self._train_frame_limit:
            terminated = True

        self._spawn_obstacle_maybe()

        obs = self._render_frame()

        return obs, reward, terminated, False, {}

    def _spawn_obstacle_maybe(self):
        if self._obstacle_cnt > max(OBSTACLE_MIN_CNT, JUMP_DURATION * self._speed):
            if self.np_random.choice(2, 1, p=[1 - self._spawn_prob, self._spawn_prob])[
                0
            ]:
                id = self.np_random.choice(len(self._assets.cactuses), 1)[0]
                self._obstacles.append(Cactus(self._assets, id))

            elif (
                self._speed >= BIRD_MIN_SPEED
                and self.np_random.choice(
                    2, 1, p=[1 - BIRD_SPAWN_PROB, BIRD_SPAWN_PROB]
                )[0]
            ):
                y = BIRD_Y_POSITIONS[self.np_random.choice(len(BIRD_Y_POSITIONS), 1)[0]]
                self._obstacles.append(Bird(self._assets, y))

            self._obstacle_cnt = 0

    def render(self):
        if self.render_mode == RenderMode.RGB:
            return self._render_frame()

    def get_state(self):
        obstacles = []
        for o in self._obstacles:
            kind = "bird" if isinstance(o, Bird) else "cactus"
            cid = None
            if kind == "cactus":
                cid = next(
                    (i for i, a in enumerate(self._assets.cactuses) if a is o._asset), 0
                )
            obstacles.append(
                (
                    kind,
                    cid,
                    o.rect.x,
                    o.rect.y,
                    o.rect.w,
                    o.rect.h,
                    o.needs_collision_check,
                )
            )
        return dict(
            frame=self._frame,
            speed=self._speed,
            spawn_prob=self._spawn_prob,
            obstacle_cnt=self._obstacle_cnt,
            agent=(self._agent.state, self._agent._jump_timer),
            track_off=self._track._track_offset_x,
            obstacles=obstacles,
            rng=self.np_random.bit_generator.state,
        )

    def set_state(self, st):
        self._frame = st["frame"]
        self._speed = st["speed"]
        self._spawn_prob = st["spawn_prob"]
        self._obstacle_cnt = st["obstacle_cnt"]
        self._agent.state, self._agent._jump_timer = st["agent"]
        self._track._track_offset_x = st["track_off"]
        self._obstacles = []
        for kind, cid, x, y, w, h, needs in st["obstacles"]:
            o = Bird(self._assets) if kind == "bird" else Cactus(self._assets, cid)
            o.rect.x, o.rect.y, o.rect.w, o.rect.h = x, y, w, h
            o.needs_collision_check = needs
            self._obstacles.append(o)
        self.np_random.bit_generator.state = st["rng"]

    def _render_frame(self) -> np.ndarray:
        canvas = pygame.Surface(WINDOW_SIZE)
        canvas.fill((255, 255, 255))

        self._track.render(canvas)
        self._agent.render(canvas)
        for o in self._obstacles:
            o.render(canvas)

        text_surface, _ = self._game_font.render(f"score: {self._frame}", (0, 0, 0))
        canvas.blit(text_surface, (10, 10))

        if self._window is not None and self._clock is not None:
            self._window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            self._clock.tick(self.metadata["render_fps"])

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()


class Wrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, k=4, image_size=(128, 64)):
        super().__init__(env)

        self.env = env
        self.k = k
        self.image_size = image_size

        obs_space = env.observation_space.shape
        assert obs_space is not None
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.k, self.image_size[1], self.image_size[0]),
            dtype=np.uint8,
        )

        self.frames: list[np.ndarray] = []
        self.stack = deque([], maxlen=self.k)

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        return np.array(
            Image.fromarray(obs).convert("L").resize(self.image_size), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        assert len(self.stack) == self.k
        return np.stack(self.stack)

    def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
        self.frames = []
        self.stack = deque([], maxlen=self.k)

        obs, _ = self.env.reset(*args, **kwargs)
        self.frames.append(obs)
        obs = self._transform(obs)

        for _ in range(self.k):
            self.stack.append(obs)

        return self._get_obs(), {}

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = False

        for _ in range(self.k):
            obs, reward, term, *_ = self.env.step(action)
            self.frames.append(obs)
            obs = self._transform(obs)

            self.stack.append(obs)

            total_reward += float(reward)
            if term:
                terminated = True
                break

        return self._get_obs(), total_reward, terminated, False, {}


register(
    id="Env-v0",
    entry_point="chrome_dino_game.envs:Env",
    max_episode_steps=300,
)
