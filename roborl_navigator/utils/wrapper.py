from typing import Any

from .converter import PandaConverter

PANDA_CONVERTER = PandaConverter()


def bullet_to_real(get_joint_angles_func):
    def wrapper(*args, **kwargs):
        joint_angles = get_joint_angles_func(*args, **kwargs)
        return PANDA_CONVERTER.bullet_to_real(joint_angles)

    return wrapper


def real_to_bullet(set_joint_angles_func):
    def wrapper(*args: Any, **kwargs: Any):
        target_joints = PANDA_CONVERTER.real_to_bullet(args[1])
        return set_joint_angles_func(args[0], target_joints)

    return wrapper
