import json

import matplotlib.pyplot as plt

file = open("performance_results_of_rl_rrt_prm.json", "r+")
results_raw = json.loads(file.read())

plt.figure(figsize=(10, 6))

results = []
for ep, res in results_raw.items():
    if 'rrt' in res:
        results.append(res)
episodes = list(range(1, len(results) + 1))

for planner in ['rrt']:
    min_times = [episode[planner]['min'] for episode in results]
    max_times = [max(episode[planner]['all'] or [1000]) for episode in results]
    mean_times = [episode[planner]['mean'] for episode in results]

    plt.plot(episodes, mean_times, label=f'RRTConnect Mean', marker='o')
    plt.fill_between(episodes, min_times, max_times, alpha=0.2, label='RRTConnect Range (min-max)')

plt.plot(episodes, [episode['rl']['total'] for episode in results], label=f'Reinforcement Learning', marker='^')


plt.xlabel('Episode')
plt.ylabel('Computation Time (ms)')
plt.title('Computation Time Range for Different Episodes')
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()
