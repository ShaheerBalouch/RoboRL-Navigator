from typing import (
    Optional,
    TypeVar,
)

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure

from roborl_navigator.utils import (
    create_directory_if_not_exists,
    get_model_directory,
)

ModelType = TypeVar('ModelType', bound=BaseAlgorithm)


class Trainer:

    def __init__(self, model: ModelType, target_step: int = 5_000, directory_path: Optional[str] = None) -> None:
        self.model = model
        if not directory_path:
            directory_path = get_model_directory()
            create_directory_if_not_exists(directory_path)

        self.save_directory = directory_path
        self.log_path = directory_path + '/logs'

        self.target_training_step = target_step
        self.logger = configure(self.log_path, ["stdout", "csv", "tensorboard"])
        self.model.set_logger(self.logger)

    def train(self) -> None:
        self.model.learn(
            total_timesteps=int(self.target_training_step),
            log_interval=10,  # episode number
        )
        self.model.save(self.save_directory + '/model')
        self.model.save_replay_buffer(self.save_directory + '/replay_buffer')
