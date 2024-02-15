import json

import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import (
    HerReplayBuffer,
    TD3,
)
from production.ros_controller import ROSController
import roborl_navigator.environment


save_path = '/assets/evaluation_results/performance_results_of_rl_rrt_prm.json'
env = gym.make("RoboRL-Navigator-Panda-ROS", orientation_task=False, distance_threshold=0.08)
m_path = '/models/roborl-navigator/TD3_Bullet_0.05_Threshold_200K/model.zip'
model = TD3.load(m_path, env=env, replay_buffer_class=HerReplayBuffer)
ros_controller = ROSController()

observation = model.env.reset()
model.predict(observation)  # to initialize

results = {}
planner_iteration = 50

for episode in range(100):
    print(f"Episode: {episode}")
    results[episode] = {'rl': {}}
    pass_episode = False

    target_position = observation["desired_goal"][:3][0]
    ros_controller.set_target_pose(target_position, np.zeros(4))
    pose = ros_controller.create_pose(target_position)

    for planner in ['rrt', 'prm']:
        planner_min = 1000
        planner_max = 0
        planner_all = []
        for _ in range(planner_iteration):
            duration = ros_controller.get_pose_goal_plan_with_duration(pose, planner)[1]
            if duration < 500:
                planner_all.append(duration)
                planner_min = min(planner_min, duration)
                planner_max = min(planner_max, duration)
            else:
                pass_episode = True
                break
        if pass_episode:
            break
        planner_mean = np.mean(np.array(planner_all))
        results[episode][planner] = {
            "min": planner_min,
            "max": planner_max,
            "mean": planner_mean,
            "all": planner_all,
        }

    rl_episode_total = 0.0
    for _ in range(50):  # 50 is episode timeout limit
        start_time = time.time()
        action = model.predict(observation)
        end_time = time.time()
        planning_time = round((end_time - start_time) * 1000)
        rl_episode_total += planning_time
        action = action[0]
        observation, reward, terminated, info = model.env.step(action)
        model.env.render()
        success = info[0].get('is_success', False)
        if terminated or success:
            print(f"RL Total Training Time of Episode: {rl_episode_total}")
            results[episode]['rl']['total'] = rl_episode_total
            results[episode]['rl']['steps'] = _
            time.sleep(3)
            break
    observation = model.env.reset()
    with open(save_path, "w") as json_file:
        json.dump(results, json_file)

print("\n\n\n")
print(results)
print("\n\n\n")
