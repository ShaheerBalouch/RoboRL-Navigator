from typing import (
    Any,
    Dict,
    Optional,
    TypeVar,
)

import numpy as np
from roborl_navigator.utils import distance, euler_to_quaternion
from roborl_navigator.simulation import Simulation
from roborl_navigator.robot import Robot

Sim = TypeVar('Sim', bound=Simulation)
Rob = TypeVar('Rob', bound=Robot)


class Reach:

    def __init__(
        self,
        sim: Sim,
        robot: Rob,
        reward_type: Optional[str] = "dense",
        distance_threshold: Optional[float] = 0.1,
        goal_range: Optional[float] = 0.3,
        orientation_task: Optional[bool] = False,
        demonstration: Optional[bool] = False,
    ) -> None:
        self.sim = sim
        self.robot = robot
        self.goal = None
        self.obstacle1_pos = None
        self.obstacle2_pos = None
        self.obstacle3_pos = None

        self.reward_type = reward_type
        self.orientation_task = orientation_task
        self.distance_threshold = distance_threshold
        self.demonstration = demonstration

        # min X can be 0.07
        self.goal_range_low = np.array([0.5 - (goal_range / 2), -goal_range / 2, 0.05])
        self.goal_range_high = np.array([0.5 + (goal_range / 2), goal_range / 2, goal_range / 2])
        self.orientation_range_low = np.array([-3, -0.8])
        self.orientation_range_high = np.array([-2, 0.4])

        self.obstacle_range = goal_range

        self.obstacle_range_low = np.array([0.5 - (self.obstacle_range / 2), -self.obstacle_range / 2, 0.05])
        self.obstacle_range_high = np.array([0.5 + (self.obstacle_range / 2), self.obstacle_range / 2, 0.05])

        self.sum_goal_reward = 0.0
        self.step_count = 0
        self.sum_dist_reward = 0.0
        self.sum_total_reward = 0.0

        with self.sim.no_rendering():
            self.create_scene()

    def create_scene(self) -> None:
        self.sim.create_scene()

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.obstacle1_pos, self.obstacle2_pos, self.obstacle3_pos = self._sample_obstacles()
        print("STEP COUNT: ", self.step_count)

        if not self.demonstration:
            self.sim.set_base_pose("target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle1", self.obstacle1_pos, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle2", self.obstacle2_pos, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle3", self.obstacle3_pos, np.array([0.0, 0.0, 0.0, 1.0]))

            if self.orientation_task:
                goal_orientation = euler_to_quaternion([self.goal[3], self.goal[4], 0])
                self.sim.set_base_pose("target_orientation_mark", self.goal[:3], goal_orientation)

    def set_goal(self, goal: np.ndarray):
        self.goal = goal

    @staticmethod
    def get_obs() -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
        if self.orientation_task:
            ee_orientation = np.array(self.robot.get_ee_orientation())[:2]
            return np.concatenate([
                ee_position,
                ee_orientation,
            ])
        return ee_position

    def _sample_goal(self) -> np.ndarray:
        position = np.random.uniform(self.goal_range_low, self.goal_range_high)
        if self.orientation_task:
            orientation = np.random.uniform(self.orientation_range_low, self.orientation_range_high)
            return np.concatenate((
                position,
                orientation,
            )).astype(np.float32)
        return position

    def _sample_obstacles(self):
        position1 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
        position2 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
        position3 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)

        while distance(position1, self.goal) < 0.05:
            position1 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
        while distance(position2, self.goal) < 0.05:
            position2 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
        while distance(position3, self.goal) < 0.05:
            position3 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)

        return position1, position2, position3

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal, self.orientation_task)
        result = np.array(d < self.distance_threshold, dtype=bool)
        return result

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any], obstacle_dist=np.array([0.1])) -> np.ndarray:
        d = distance(achieved_goal, desired_goal, self.orientation_task)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            goal_reward = -d.astype(np.float32)

            # dist_x, dist_y, dist_z = obstacle_dist
            # reward_x = np.clip(-(0.1-dist_x)/3, -0.1, 0.0)
            # reward_y = np.clip(-(0.1-dist_y)/3, -0.1, 0.0)
            # reward_z = np.clip(-(0.1-dist_z)/3, -0.1, 0.0)
            # dist_reward = (reward_x + reward_y + reward_z).astype(np.float32)

            dist_reward = np.clip(-(0.1-obstacle_dist).astype(np.float32), -0.1, 0.0)

            reward = dist_reward + goal_reward
            if goal_reward.shape == ():
                self.step_count += 1

            return reward

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()
