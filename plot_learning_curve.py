import pandas as pd
import matplotlib.pyplot as plt

mode = "len_max"  # len_max or len

# Load the CSV file
if mode == "len_max":
    file_path = 'data/exp_len_max.csv'
elif mode == "len":
    file_path = 'data/exp_len_average.csv'

data = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
window_size = 100
data_colmns = [col for col in data.columns if col.endswith('len/' + mode)]
for col in data_colmns:
    filtered_data = data.dropna(subset=[col])
    smoothed_values = filtered_data[col].rolling(window=window_size, min_periods=1).mean()
    plt.plot(filtered_data['global_step'], smoothed_values, label=col, linestyle='-', alpha=0.9)

plt.plot([0, 5 * 10 ** 9], [10_000, 10_000], c="k", alpha=0.5, linestyle="--")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Training step', fontsize=20)
if mode == "len_max":
    plt.ylabel('Maximum episode length', fontsize=20)
    # plt.savefig("len_max.pdf")
elif mode == "len":
    plt.ylabel('Average episode length', fontsize=20)
    # plt.savefig("len.pdf")
plt.show()
