import json
import numpy as np
import matplotlib.pyplot as plt

video_points = [467, 615, 950, 1203]

with open("data/progress.json", "r") as outfile:
    data_json = json.load(outfile)

max_step = data_json["step"][-1]

plt.figure(figsize=(12, 7))
plt.subplot(411)
plt.plot(data_json["step"], np.array(data_json["health"])/9, c="r", alpha=0.5, label="health")
plt.plot(data_json["step"], np.array(data_json["hunger"])/9, c="brown", alpha=0.5, label="food")
plt.plot(data_json["step"], np.array(data_json["thirst"])/9, c="b", alpha=0.5, label="water")
plt.plot(data_json["step"], np.array(data_json["energy"])/9, c="g", alpha=0.5, label="energy")
plt.xlim([0, max_step])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(loc='center right', facecolor="white", framealpha=1)

plt.subplot(412)
plt.plot(data_json["step"], data_json["eat_cow"], c="brown", alpha=0.7, label="eat cow")
plt.plot(data_json["step"], data_json["drink"], c="b", alpha=0.7, label="drink")
plt.xlim([0, max_step])
plt.xticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(loc='center right', facecolor="white", framealpha=1)

plt.subplot(413)
sleep_actions = np.array(data_json["action"], dtype=np.int64) == 6 # sleep action
plt.plot(data_json["step"], sleep_actions, c="r", alpha=0.7, label="sleep actions", zorder=0)
plt.plot(data_json["step"], data_json["wake_up"], c="g", alpha=0.8, label="wake up", zorder=0)
plt.plot(data_json["step"], np.array(data_json["energy"])/9, "g--", alpha=0.5, label="energy")
plt.plot(data_json["step"], data_json["daylight"], c="k", alpha=0.7, label="dayliht", zorder=1)
plt.scatter(video_points, [data_json["daylight"][p] for p in video_points], c="k", s=50, zorder=2)

plt.xlim([0, max_step])
plt.xticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(loc='center right', facecolor="white", framealpha=1)

plt.subplot(414)
plt.plot(data_json["step"], data_json["defeat_zombie"], c="purple", alpha=0.8, label="defeat zombie")
plt.xlim([0, max_step])
plt.xticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(loc='center right', facecolor="white", framealpha=1)

plt.tight_layout()
plt.pause(0.1)
#plt.savefig("progress.pdf")

# save video frames
video_frames = np.load(file="data/video_frame.npy")
for i in video_points:
    plt.figure()
    plt.imshow(video_frames[i])
    plt.axis(False)
    # plt.savefig(f"data/frame_{i}.pdf")

plt.show()
