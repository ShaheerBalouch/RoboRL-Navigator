import time

from stable_baselines3 import (
    HerReplayBuffer,
    TD3,
)
import gymnasium as gym
import roborl_navigator.environment


env = gym.make(
    "RoboRL-Navigator-Panda-ROS",
    orientation_task=True,
    distance_threshold=0.08,
)

model = TD3.load(
    '/home/juanhernandezvega/dev/RoboRL-Navigator/models/roborl-navigator/TD3_orientation_200K/model.zip',
    env=env,
    replay_buffer_class=HerReplayBuffer,
)

observation = model.env.reset()

# Evaluate the agent
for episode in range(1_000):
    for _ in range(50):
        action = model.predict(observation)[0]
        observation, reward, terminated, info = model.env.step(action)
        model.env.render()
        success = info[0].get('is_success', False)
        if terminated or success:
            time.sleep(3)
            break
    observation = model.env.reset()
