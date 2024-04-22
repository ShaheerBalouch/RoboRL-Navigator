import numbers

import numpy as np
import pybullet as p
import pybullet_data
import time
from stable_baselines3 import (
    HerReplayBuffer,
    TD3,
)

from roborl_navigator.environment.env_panda_bullet import PandaBulletEnv
print()
path = '../models/roborl-navigator/APR_07_16/model.zip'

env = PandaBulletEnv(
    orientation_task=False,
    distance_threshold=0.025,
    goal_range=0.02
)
model = TD3.load(
    path,
    env=env,
    replay_buffer_class=HerReplayBuffer,
)

sim = env.sim
target_pose_array = np.array([0.7, 0.0, 0.1])

# sim.remove_model("obstacle1")
# cube = p.loadURDF("cube.urdf", [0.75, 0, 0.1], globalScaling=0.05)

observation = env.reset(options={"goal": np.array(target_pose_array[:3]).astype(np.float32)})[0]
sim.set_base_pose("target", target_pose_array, np.array([0, 0, 0, 1]))
sim.set_base_pose("obstacle1", np.array([0.65, 0.0, 0.05]), np.array([0, 0, 0, 1]))
# input("Press Enter to continue...")

for step in range(50):
    action = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(np.array(action[0]).astype(np.float32))
    if terminated or info.get('is_success', False):
        print("Reached destination!")
        time.sleep(20)
        break

