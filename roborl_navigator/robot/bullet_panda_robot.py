import numpy as np

from roborl_navigator.simulation.bullet import BulletSim
from roborl_navigator.robot.base_robot import Robot
from roborl_navigator.utils import (
    bullet_to_real,
    real_to_bullet,
)


class BulletPanda(Robot):

    def __init__(self, sim: BulletSim, orientation_task: bool = False) -> None:
        super().__init__(sim, orientation_task)
        self.joint_indices = np.array([0, 1, 2, 3, 4, 5, 6])
        self.joint_forces = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0])
        self.body_name = "panda"
        self.ee_link = 11
        with self.sim.no_rendering():
            self.load_robot("franka_panda/panda.urdf", np.zeros(3))

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_orientation(self) -> np.ndarray:
        return np.array(self.get_link_orientation(self.ee_link)).astype(np.float32)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)"""
        return self.sim.get_link_position(self.body_name, link)

    def get_link_orientation(self, link: int) -> np.ndarray:
        ori = self.sim.get_link_orientation(self.body_name, link)
        return self.sim.physics_client.getEulerFromQuaternion(ori)

    def get_link_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (vx, vy, vz)"""
        return self.sim.get_link_velocity(self.body_name, link)

    # Wrapper not applied because it is a sub-method
    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint"""
        return self.sim.get_joint_angle(self.body_name, joint)

    @bullet_to_real
    def get_joint_angles(self) -> np.ndarray:
        return np.array([self.get_joint_angle(joint=i) for i in range(7)])

    def get_target_arm_angles(self, joint_actions: np.ndarray) -> np.ndarray:
        joint_actions = joint_actions * 0.05  # limit maximum change in position
        return self.get_joint_angles() + joint_actions

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        arm_joint_ctrl = action[:7]
        target_arm_angles = self.get_target_arm_angles(arm_joint_ctrl)
        self.control_joints(target_arm_angles)

    @real_to_bullet
    def set_joint_angles(self, joint_values: np.ndarray) -> None:
        """Set the joint position of a body. Can induce collisions."""
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices, angles=joint_values)

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    @real_to_bullet
    def control_joints(self, joint_values: np.ndarray) -> None:
        """Control the joints of the robot."""
        self.sim.control_joints(
            body=self.body_name,
            joints=self.joint_indices,
            target_angles=joint_values,
            forces=self.joint_forces,
        )

    def load_robot(self, file_name: str, base_position: np.ndarray) -> None:
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            useFixedBase=True,
        )
