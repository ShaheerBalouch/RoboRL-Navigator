import gymnasium as gym

from stable_baselines3 import (
    DDPG,
    HerReplayBuffer,
    SAC,
    TD3,
)
from train.trainer import Trainer
import roborl_navigator.environment

env = gym.make("RoboRL-Navigator-Panda-ROS", orientation_task=True, distance_threshold=0.05)
model = TD3(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)
trainer = Trainer(model=model, target_step=50_000)

trainer.train()
