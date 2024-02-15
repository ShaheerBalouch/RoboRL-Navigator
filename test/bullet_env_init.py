import numpy as np

from roborl_navigator.environment.env_panda_bullet import PandaBulletEnv

"""
TEST Bullet Environment Initialization
"""

env = PandaBulletEnv(orientation_task=True, render_mode="human")
env.reset()

action = np.ones(7)
observation, reward, terminated, truncated, info = env.step(action)
action = np.zeros(7)
for i in range(1_000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if i % 50 == 0:
        env.reset()
