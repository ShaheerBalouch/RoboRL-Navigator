import csv
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

path = '../models/roborl-navigator/APR_30_1/model.zip'

env = PandaBulletEnv(
    orientation_task=False,
    distance_threshold=0.025,
    goal_range=0.2
)
model = TD3.load(
    path,
    env=env,
    replay_buffer_class=HerReplayBuffer,
)

number_of_successes = 0
number_of_collisions = 0
number_of_timeouts = 0

sim = env.sim

filename = "random_goals_005_00.csv"

with open(filename, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:

        goal_position = np.array([float(row[0]), float(row[1]), float(row[2])])
        obs_pos_1 = np.array([float(row[3]), float(row[4]), float(row[5])])
        obs_pos_2 = np.array([float(row[6]), float(row[7]), float(row[8])])
        obs_pos_3 = np.array([float(row[9]), float(row[10]), float(row[11])])

        target_pose_array = goal_position

        observation = env.reset(options={"goal": np.array(target_pose_array[:3]).astype(np.float32)})[0]
        sim.set_base_pose("target", target_pose_array, np.array([0, 0, 0, 1]))

        sim.set_base_pose("obstacle1", obs_pos_1, np.array([0, 0, 0, 1]))
        sim.set_base_pose("obstacle2", obs_pos_2, np.array([0, 0, 0, 1]))
        sim.set_base_pose("obstacle3", obs_pos_3, np.array([0, 0, 0, 1]))

        for step in range(50):
            action = model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(np.array(action[0]).astype(np.float32))
            if terminated:
                if info.get('is_success'):
                    number_of_successes += 1

                break
        if not info.get('is_success'):

            if info.get('is_collision'):
                number_of_collisions += 1
            else:
                number_of_timeouts += 1

print("NUMBER OF SUCCESSES: ", number_of_successes)
print("NUMBER OF COLLISIONS: ", number_of_collisions)
print("NUMBER OF TIMEOUTS: ", number_of_timeouts)