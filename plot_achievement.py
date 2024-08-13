import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


jsonl_file_path = "data/stats1000.jsonl"


def gmean_crafter(v):
    n = 22
    return np.exp(np.sum(np.log(v + 1)) / n) - 1


colors = ['#288c66', '#bf3217', '#de9f42', '#6a554d',]
pos = [-1.5, -0.5, 0.5, 1.5]

x = np.arange(22)
width = 0.6
fig, ax = plt.subplots(figsize=(8, 4))
data_jsonl = []
with open(jsonl_file_path, 'r') as file:
    for line in file:
        data_jsonl.append(json.loads(line))

stats = {k: [] for k in data_jsonl[0].keys()}
for unused in {"length", "reward"}:
    stats.pop(unused)

for data in data_jsonl:
    for k in stats.keys():
        is_achieved = np.array(data[k]) > 0
        stats[k].append(100 * is_achieved)

achievement_name = stats.keys()
df = pd.DataFrame(stats)

means = df.mean()
print("Homeostasis:", gmean_crafter(means.values))

ax.bar(x, means, width, label="Homeostasis", color=colors[0])
plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.ylim([0.01, 100])
ax.set_yticks([0.01, 0.1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100'])
ax.set_ylabel('Success rate (%)')
ax.set_xticks(x)
ax.tick_params(axis='x', which='both', length=0)
for i in x:
    ax.plot([i - width * 0.6, i + width * 0.6], [0.01, 0.01], color='black', linewidth=1)

ax.set_xticklabels([
    "Collect Coal",
    "Collect Diamond",
    "Collect Drink",
    "Collect Iron",
    "Collect Sapling",
    "Collect Stone",
    "Collect Wood",
    "Defeat Skeleton",
    "Defeat Zombie",
    "Eat Cow",
    "Eat Plant",
    "Make Iron Pickaxe",
    "Make Iron Sword",
    "Make Stone Pickaxe",
    "Make Stone Sword",
    "Make Wood Pickaxe",
    "Make Wood Sword",
    "Place Furnace",
    "Place Plant",
    "Place Stone",
    "Place Table",
    "Wake Up"
], rotation=45)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
#plt.savefig(f"achievement_prob.pdf")
plt.show()
