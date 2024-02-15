from enum import Enum


class PlannerResult(Enum):
    SUCCESS = 1
    COLLISION = 2
    MOVEIT_ERROR = 3
