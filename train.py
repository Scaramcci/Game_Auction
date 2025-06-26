import gymnasium as gym
from stable_baselines3 import PPO
from env import InsiderKyleEnv

# 配置环境参数
T = 10
sigma_u = 1.0
sigma_v = 1.0
lambda_val = 0.5
max_action = 5.0
seed = 42

# 创建环境实例并设定随机种子
env = InsiderKyleEnv(T=T, sigma_u=sigma_u, sigma_v=sigma_v, lambda_val=lambda_val, max_action=max_action, seed=seed)

# 使用稳定基线库初始化 PPO 模型（多层感知机策略）
model = PPO("MlpPolicy", env, verbose=1)

# 开始训练智能体（设定训练的总时间步，如 100000）
total_timesteps = 100000
model.learn(total_timesteps=total_timesteps)

# 保存训练好的模型策略
model.save("insider_policy")
print("训练结束，模型已保存为 insider_policy.zip")