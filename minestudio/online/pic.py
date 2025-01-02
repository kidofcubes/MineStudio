import matplotlib.pyplot as plt

# 读取保存的结果
timesteps = []
mean_rewards = []
variances = []

with open("inference_results.txt", "r") as f:
    for line in f:
        parts = line.strip().split(", ")
        timestep = int(parts[0].split(": ")[1])
        mean_reward = float(parts[1].split(": ")[1])
        variance = float(parts[2].split(": ")[1])
        timesteps.append(timestep)
        mean_rewards.append(mean_reward)
        variances.append(variance)

# 转换为 numpy 数组以便绘图
timesteps = np.array(timesteps)
mean_rewards = np.array(mean_rewards)
variances = np.array(variances)

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(timesteps, mean_rewards, label="Mean Reward", marker="o")
plt.fill_between(
    timesteps,
    mean_rewards - np.sqrt(variances),
    mean_rewards + np.sqrt(variances),
    alpha=0.3,
    label="Standard Deviation"
)
plt.xlabel("Timestep")
plt.ylabel("Mean Reward")
plt.title("Mean Reward vs Timestep")
plt.legend()
plt.grid()
plt.savefig("reward_plot.png")
plt.show()
