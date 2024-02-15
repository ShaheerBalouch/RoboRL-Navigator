import numpy as np
import time
from stable_baselines3 import (
    HerReplayBuffer,
    TD3,
)

from roborl_navigator.environment.env_panda_ros import PandaROSEnv
from roborl_navigator.robot.ros_panda_robot import ROSRobot
from roborl_navigator.simulation.ros import ROSSim

from ros_controller import ROSController

env = PandaROSEnv(
    orientation_task=False,
    distance_threshold=0.03,
    demonstration=True,
    real_robot=True,
)
model = TD3.load(
    '/home/franka/dev/RoboRL-Navigator/models/roborl-navigator/TD3_Bullet_0.05_Threshold_200K/model.zip',
    env=env,
    replay_buffer_class=HerReplayBuffer,
)
sim = ROSSim(orientation_task=False)
robot = ROSRobot(sim=sim, orientation_task=False, real_robot=False)
ros_controller = ROSController(real_robot=True)

# Open the gripper
ros_controller.hand_open()

target_pose_array = [0.8, -0.1, 0.11]

print(f"Desired Goal: {target_pose_array[:3]}")

ros_controller.go_to_home_position()
# Go To Trained Starting Point
observation = env.reset(options={"goal": np.array(target_pose_array[:3]).astype(np.float32)})[0]

for _ in range(10):
    action = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(np.array(action[0]).astype(np.float32))
    if terminated or info.get('is_success', False):
        print("Reached destination!")
        break

# Close Gripper
ros_controller.hand_grasp()

ros_controller.go_to_release_position()
time.sleep(1.5)
ros_controller.hand_open()
