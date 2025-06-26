import gymnasium as gym
from stable_baselines3 import PPO
from env import InsiderKyleEnv

# 配置环境参数
T = 10
sigma_u = 0.8  # 降低噪声交易方差，减少价格波动
sigma_v = 1.2  # 增加真值先验方差，提供更多信息价值
lambda_val = 0.3  # 降低做市商定价系数，减少价格冲击
max_action = 3.0  # 降低最大动作范围，避免过度激进策略
seed = 42

# 创建环境实例并设定随机种子
env = InsiderKyleEnv(T=T, sigma_u=sigma_u, sigma_v=sigma_v, lambda_val=lambda_val, max_action=max_action, seed=seed)

# 使用稳定基线库初始化 PPO 模型（多层感知机策略）
# 调整学习率和其他超参数以提高学习效果
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10)

# 开始训练智能体（增加训练步数以获得更好的策略）
total_timesteps = 200000
model.learn(total_timesteps=total_timesteps)

# 保存训练好的模型策略
model.save("insider_policy")
print("训练结束，模型已保存为 insider_policy.zip")