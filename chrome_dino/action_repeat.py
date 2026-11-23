"""Action-repeat wrapper used by policy training and evaluation."""

import gymnasium as gym

from chrome_dino_game import envs  # noqa: F401 (registers the gym environment)


class RepeatEnv(gym.Wrapper):
    """Repeat an action, sum rewards, and return the final frame."""

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k

    def step(self, action):
        total, obs, term, trunc, info = 0.0, None, False, False, {}
        for _ in range(self.k):
            obs, r, term, trunc, info = self.env.step(action)
            total += float(r)
            if term:
                break
        return obs, total, term, trunc, info


def make_repeat_pixel_envs(n: int, game_mode: str = "train", k: int = 4):
    gym_envs, raw_envs = [], []
    for _ in range(n):
        e = gym.make(
            "Env-v0",
            render_mode="rgb_array",
            game_mode=game_mode,
            train_frame_limit=99999,
            disable_env_checker=True,
        )
        gym_envs.append(RepeatEnv(e, k=k))
        raw_envs.append(e.unwrapped)
    return gym_envs, raw_envs


def frames_to_decisions(max_steps: int, k: int) -> int:
    return max(1, max_steps // k)
