from typing import (
    Dict,
    Any,
    Optional,
    Tuple,
)

import numpy as np

from gymnasium.utils import seeding

from roborl_navigator.environment import BaseEnv
from roborl_navigator.simulation.ros.ros_sim import ROSSim
from roborl_navigator.robot.ros_panda_robot import ROSRobot
from roborl_navigator.task.reach_task import Reach


class PandaROSEnv(BaseEnv):

    def __init__(
        self,
        orientation_task: bool = False,
        distance_threshold: float = 0.05,
        goal_range: float = 0.3,
        demonstration: bool = False,
        real_robot: bool = False,
    ) -> None:
        self.sim = ROSSim(orientation_task=orientation_task, demonstration=demonstration)
        self.robot = ROSRobot(self.sim, orientation_task=orientation_task, real_robot=real_robot)
        self.task = Reach(
            self.sim,
            self.robot,
            reward_type="dense",
            orientation_task=orientation_task,
            distance_threshold=distance_threshold,
            demonstration=demonstration,
            goal_range=goal_range,
        )
        self.demonstration = demonstration
        super().__init__()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if self.demonstration and options and "goal" in options:
            self.task.set_goal(options["goal"])
        else:
            super().reset(seed=seed, options=options)
            self.task.np_random, seed = seeding.np_random(seed)
            with self.sim.no_rendering():
                self.robot.reset()
                self.task.reset()

        observation = self._get_obs()
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        truncated = self.robot.set_action(action)
        observation = self._get_obs()
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        info = {"is_success": terminated}
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    def render(self) -> Optional[np.ndarray]:
        pass
