import numpy as np
import csv
from roborl_navigator.utils import distance

filename = "random_goals_003.csv"
num_rows = 500

goal_range = 0.2
goal_range_low = np.array([0.5 - (goal_range / 2), -goal_range / 2, 0.05])
goal_range_high = np.array([0.5 + (goal_range / 2), goal_range / 2, goal_range / 2])

obstacle_range = goal_range
obstacle_range_low = np.array([0.5 - (obstacle_range / 2), -obstacle_range / 2, 0.05])
obstacle_range_high = np.array([0.5 + (obstacle_range / 2), obstacle_range / 2, 0.05])

dist_to_goal = 0.03

with open(filename, "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    for i in range(num_rows):
        goal_position = np.random.uniform(goal_range_low, goal_range_high)

        obs_pos_1 = np.random.uniform(obstacle_range_low, obstacle_range_high)
        obs_pos_2 = np.random.uniform(obstacle_range_low, obstacle_range_high)
        obs_pos_3 = np.random.uniform(obstacle_range_low, obstacle_range_high)

        while distance(obs_pos_1, goal_position) < dist_to_goal:
            obs_pos_1 = np.random.uniform(obstacle_range_low, obstacle_range_high)
        while distance(obs_pos_2, goal_position) < dist_to_goal:
            obs_pos_2 = np.random.uniform(obstacle_range_low, obstacle_range_high)
        while distance(obs_pos_3, goal_position) < dist_to_goal:
            obs_pos_3 = np.random.uniform(obstacle_range_low, obstacle_range_high)

        random_arrays = np.concatenate((goal_position, obs_pos_1, obs_pos_2, obs_pos_3))

        csv_writer.writerow(random_arrays)
