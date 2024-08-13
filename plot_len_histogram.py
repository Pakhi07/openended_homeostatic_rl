# Load the JSONL file and extract the "length" data
import json

from matplotlib import pyplot as plt

jsonl_file_path = "data/stats1000.jsonl"

length_data = []

with open(jsonl_file_path, 'r') as file:
    for line in file:
        record = json.loads(line)
        if 'length' in record:
            length_data.append(record['length'])

plt.figure(figsize=(8, 6))
plt.hist(length_data, range=(0, 10_000), edgecolor=(1, 1, 1))
plt.xlabel('Episode length', fontsize=20)
plt.ylabel('Frequency', fontsize=20)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
# plt.savefig("len_histogram.pdf")
