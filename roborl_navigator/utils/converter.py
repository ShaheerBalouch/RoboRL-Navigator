import numpy as np


class PandaConverter:

    def __init__(self):
        # Joint Degree Wrapper
        self.real_panda_limits = (
            [-2.89, 2.89],
            [-1.76, 1.76],
            [-2.89, 2.89],
            [-3.07, 0.06],
            [-2.89, 2.89],
            [-0.02, 3.75],
            [-2.89, 2.89],
        )

        self.bullet_panda_limits = (
            [-2.967, 2.967],
            [-1.83, 1.83],
            [-2.967, 2.967],
            [-3.14, 0.0],
            [-2.967, 2.967],
            [-0.087, 3.822],
            [-2.967, 2.967],
        )

    @staticmethod
    def map(value, from_min, from_max, to_min, to_max, round_decimal=2):
        clamped_value = max(from_min, min(value, from_max))
        return round(((clamped_value - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min, round_decimal)

    def zip(self, values, limits, obs_range):
        return np.array(
            [
                max(
                    obs_range[0], min(obs_range[1], self.map(v, limits[i][0], limits[i][1], obs_range[0], obs_range[1]))
                )
                for i, v in enumerate(values)
            ]
        )

    def unzip(self, values, limits, obs_range):
        return np.array(
            [self.map(v, obs_range[0], obs_range[1], limits[i][0], limits[i][1]) for i, v in enumerate(values)]
        )

    @staticmethod
    def map_value(value, from_range, to_range):
        from_min, from_max = from_range
        to_min, to_max = to_range

        # Ensure the value is within the from_range
        value = max(from_min, min(from_max, value))

        # Map the value to the new range
        mapped_value = ((value - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min
        return mapped_value

    def bullet_to_real(self, joint_values):
        mapped_values = []
        for input_value, from_range, to_range in zip(joint_values, self.bullet_panda_limits, self.real_panda_limits):
            mapped_value = self.map_value(input_value, from_range, to_range)
            mapped_values.append(mapped_value)
        return np.array(mapped_values)

    def real_to_bullet(self, joint_values):
        mapped_values = []
        for input_value, from_range, to_range in zip(joint_values, self.real_panda_limits, self.bullet_panda_limits):
            mapped_value = self.map_value(input_value, from_range, to_range)
            mapped_values.append(mapped_value)
        return np.array(mapped_values)
