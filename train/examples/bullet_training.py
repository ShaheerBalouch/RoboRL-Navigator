import gymnasium as gym

from stable_baselines3 import (
    DDPG,
    HerReplayBuffer,
    SAC,
    TD3,
)
from train.trainer import Trainer
import roborl_navigator.environment

env = gym.make(
    "RoboRL-Navigator-Panda-Bullet",
    render_mode="human",
    orientation_task=False,
    distance_threshold=0.05,
    goal_range=0.2,
)

model = TD3(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)

trainer = Trainer(model=model, target_step=50_000)

trainer.train()
