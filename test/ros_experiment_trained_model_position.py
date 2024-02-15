import time

from stable_baselines3 import (
    HerReplayBuffer,
    TD3,
)
import gymnasium as gym
import roborl_navigator.environment


env = gym.make("RoboRL-Navigator-Panda-ROS", orientation_task=False, distance_threshold=0.05)

model = TD3.load(
    '/home/juanhernandezvega/dev/RoboRL-Navigator/models/roborl-navigator/TD3_Bullet_0.05_Threshold_200K/model.zip',
    env=env,
    replay_buffer_class=HerReplayBuffer,
)

observation = model.env.reset()
# Evaluate the agent
for episode in range(1_000):
    episode_total = 0.0
    for _ in range(50):
        start_time = time.time()
        action = model.predict(observation)
        end_time = time.time()
        planning_time = round((end_time - start_time) * 1000)
        episode_total += planning_time
        action = action[0]
        observation, reward, terminated, info = model.env.step(action)
        model.env.render()
        success = info[0].get('is_success', False)
        if terminated or success:
            print(f"Total Training Time of Episode {_}: {episode_total} ms")
            time.sleep(3)
            break
    observation = model.env.reset()
