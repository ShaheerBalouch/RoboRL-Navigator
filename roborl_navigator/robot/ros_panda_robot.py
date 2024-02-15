from collections import deque
from typing import Optional

import numpy as np

from roborl_navigator.robot.base_robot import Robot
from roborl_navigator.simulation.ros.ros_sim import ROSSim
from roborl_navigator.utils import PlannerResult

try:
    import moveit_commander
    from tf.transformations import euler_from_quaternion, quaternion_from_euler
except ImportError:
    print("ROS Packages are not initialized!")


class ROSRobot(Robot):

    def __init__(self, sim: ROSSim, orientation_task: bool = False, real_robot: bool = False) -> None:
        super().__init__(sim, orientation_task)
        self.real_robot = real_robot
        self.robot_name = "fr3" if real_robot else "panda"
        self.move_group = moveit_commander.MoveGroupCommander(self.robot_name + "_manipulator")
        self.status_queue = deque(maxlen=5)

    def get_ee_position(self) -> np.ndarray:
        position = self.move_group.get_current_pose().pose.position
        return np.array([
            position.x,
            position.y,
            position.z,
        ]).astype(np.float32)

    def get_ee_orientation(self) -> np.ndarray:
        orientation = self.move_group.get_current_pose().pose.orientation
        return np.array(
            euler_from_quaternion([
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ])
        ).astype(np.float32)

    def get_ee_velocity(self) -> np.ndarray:
        return np.zeros(3)

    def get_target_arm_angles(self, joint_actions: np.ndarray) -> np.ndarray:
        joint_actions = joint_actions * 0.05
        return self.get_joint_angles() + joint_actions

    def get_joint_angles(self) -> np.ndarray:
        return np.array(self.move_group.get_current_joint_values())

    def set_action(self, action: np.ndarray) -> Optional[bool]:
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        arm_joint_ctrl = action[:7]
        target_arm_angles = self.get_target_arm_angles(arm_joint_ctrl)

        stuck = None
        result = self.control_joints(target_arm_angles)
        self.status_queue.append(result)

        if result in [PlannerResult.MOVEIT_ERROR, PlannerResult.COLLISION]:
            stuck = self.stuck_check()
            if not stuck:
                target_arm_angles = self.get_target_arm_angles(arm_joint_ctrl / 2.0)
                result = self.control_joints(target_arm_angles)

        if result != PlannerResult.SUCCESS:
            if stuck:
                return True
            else:
                return False
        return False

    def set_joint_angles(self, joint_values: np.ndarray) -> PlannerResult:
        return self.control_joints(joint_values)

    def set_joint_neutral(self) -> None:
        self.set_joint_angles(self.neutral_joint_values)

    def control_joints(self, joint_values: np.ndarray) -> PlannerResult:
        if self.real_robot:
            joint_values = joint_values.tolist()
            self.move_group.go(joint_values, True)
            try:
                self.move_group.go(joint_values, True)
            except moveit_commander.MoveItCommanderException:
                return PlannerResult.MOVEIT_ERROR
            self.move_group.stop()
            return PlannerResult.SUCCESS
        else:
            try:
                success, plan, _, _ = self.move_group.plan(joint_values)
            except moveit_commander.MoveItCommanderException:
                return PlannerResult.MOVEIT_ERROR
            if not success:
                return PlannerResult.COLLISION
            try:
                self.move_group.go(joint_values, True)
            except moveit_commander.MoveItCommanderException:
                return PlannerResult.MOVEIT_ERROR
            self.move_group.stop()
            return PlannerResult.SUCCESS

    # ROS Specific
    def stuck_check(self) -> bool:
        if not self.status_queue:
            return False
        first_value = self.status_queue[0]
        if first_value == PlannerResult.SUCCESS:
            return False
        if len(self.status_queue) < self.status_queue.maxlen:
            return False
        for value in self.status_queue:
            if value != first_value:
                return False
        return True
