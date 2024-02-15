import matplotlib.pyplot as plt
import json

file = open('performance_results_of_rl_rrt_prm_3.json', 'r+')
raw_data = json.loads(file.read())
data = []
for i in raw_data:
    if i['rrt'] < 1000 or i['prm'] < 1000:
        data.append(i)

rrt_values = [entry["rrt"] for entry in data]
prm_values = [entry["prm"] for entry in data]
rl_values = [entry["rl"] for entry in data]
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(data) + 1), rrt_values, marker='o', label="RRT")
plt.plot(range(1, len(data) + 1), prm_values, marker='s', label="PRM")
plt.plot(range(1, len(data) + 1), rl_values, marker='^', label="RL")

plt.xlabel("Episode")
plt.ylabel("Computation Time (ms)")
x_ticks = range(1, len(data) + 1, 4)
plt.xticks(x_ticks)

plt.legend()
# plt.title("Computation Times of PRM, RRT, and RL")

plt.show()
