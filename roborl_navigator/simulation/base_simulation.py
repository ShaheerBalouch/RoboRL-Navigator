from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    Iterator,
    Optional,
)

import numpy as np


class Simulation(ABC):

    def __init__(self, render_mode: str = "rgb_array", n_substeps: int = 20) -> None:
        self.render_mode = render_mode
        self.n_substeps = n_substeps
        self.timestep = 1.0 / 500
        self._bodies_idx = {}

    @property
    def dt(self):
        """Timestep"""
        return self.timestep * self.n_substeps

    def step(self) -> None:
        pass

    def close(self) -> None:
        pass

    def render(
        self,
        width: int = 720,
        height: int = 480,
        target_position: Optional[np.ndarray] = None,
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        pass

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        yield

    @abstractmethod
    def create_scene(self) -> None:
        pass

    @abstractmethod
    def create_sphere(self, position: np.ndarray) -> None:
        pass

    @abstractmethod
    def create_orientation_mark(self, position: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_base_pose(self, body: str, position: np.ndarray, orientation: np.ndarray) -> None:
        pass
