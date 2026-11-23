"""Scripted policy used to record simulator datasets."""

from common import envs


LOW, MID, HIGH = envs.BIRD_Y_POSITIONS[:3]
DINO_X = 50


def oracle_action(
    raw_env,
    jump_ttc: float = 8.0,
    duck_lo: float = -2.0,
    duck_hi: float = 6.0,
    jump_lo: float = 0.0,
):
    pending = [o for o in raw_env._obstacles if o.needs_collision_check]
    if not pending:
        return envs.Action.STAND
    obstacle = min(pending, key=lambda o: o.rect.x)
    ttc = (obstacle.rect.x - DINO_X) / max(raw_env._speed, 1)
    if isinstance(obstacle, envs.Bird):
        if obstacle.rect.y == MID:
            return envs.Action.DUCK if duck_lo < ttc < duck_hi else envs.Action.STAND
        if obstacle.rect.y == LOW and jump_lo <= ttc <= jump_ttc:
            return envs.Action.JUMP
        return envs.Action.STAND
    return envs.Action.JUMP if jump_lo <= ttc <= jump_ttc else envs.Action.STAND
