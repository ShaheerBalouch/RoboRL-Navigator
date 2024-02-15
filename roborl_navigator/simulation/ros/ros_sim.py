import numpy as np
import os
import rospy
import sys
from typing import (
    Any,
    Optional,
)

import moveit_commander
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import (
    SetModelState,
    SpawnModel,
)
from geometry_msgs.msg import Pose

from roborl_navigator.simulation import Simulation
from roborl_navigator.utils import euler_to_quaternion


class ROSSim(Simulation):
    """ROSSim basically represents Gazebo Simulation"""

    def __init__(self, orientation_task: bool = False, demonstration: bool = False) -> None:
        super().__init__()
        self.orientation_task = orientation_task
        self.demonstration = demonstration

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_controller', anonymous=True)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()

        # Object Manager
        self.model_paths = {
            "target": "target.xml",
            "target_orientation_mark": "target_orientation_mark.xml",
            "obstacle_object": "obstacle_object_base.xml",
            "aim_sphere": "small_aim_sphere.xml",
        }
        self.models = {}
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def step(self) -> None:
        return None

    def close(self) -> None:
        return None

    def render(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        return None

    def create_scene(self) -> None:
        # Create a ground collision object in the RVIZ
        if not self.demonstration:
            self.create_object("target", np.zeros(3), np.zeros(3))
            if self.orientation_task:
                self.create_object("target_orientation_mark", np.zeros(3), np.zeros(3))

    def create_sphere(self, position: np.ndarray) -> None:
        pass

    def create_orientation_mark(self, position: np.ndarray) -> None:
        pass

    def set_base_pose(self, body: str, position: np.ndarray, orientation: np.ndarray) -> None:
        rospy.wait_for_service('/gazebo/set_model_state')
        for i in range(100):  # To avoid latency bug
            state_msg = ModelState()
            state_msg.model_name = body
            state_msg.pose.position.x = position[0]
            state_msg.pose.position.y = position[1]
            state_msg.pose.position.z = position[2]
            state_msg.pose.orientation.x = orientation[0]
            state_msg.pose.orientation.y = orientation[1]
            state_msg.pose.orientation.z = orientation[2]
            state_msg.pose.orientation.w = orientation[3]
            self.set_model_state_proxy(state_msg)

    def create_object(self, body: str, position: np.ndarray, orientation: np.ndarray) -> None:
        self.retrieve_model(body)
        if body not in self.models:
            print(f"Model name {body} not in self.models cache.")
            return None
        model = self.models[body]

        model_state = Pose()
        model_state.position.x = position[0]
        model_state.position.y = position[1]
        model_state.position.z = position[2]

        if orientation is not None:
            quaternion = euler_to_quaternion(orientation)
            model_state.orientation.x = quaternion[0]
            model_state.orientation.y = quaternion[1]
            model_state.orientation.z = quaternion[2]
            model_state.orientation.w = quaternion[3]

        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_model(body, str(model), "", model_state, "world")

    def retrieve_model(self, model_name: str) -> Optional[bool]:
        if model_name not in self.model_paths:
            print(f"Model name ({model_name}) not in self.model_paths")
            return False
        if model_name not in self.models:
            project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '', '../..', '..'))
            model_path = os.path.join(project_dir, 'assets', 'object_models', self.model_paths[model_name])
            self.models[model_name] = open(model_path, "r+").read()
