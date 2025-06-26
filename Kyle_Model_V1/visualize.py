import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from stable_baselines3 import PPO
from env import InsiderKyleEnv

# 加载训练好的策略模型
model = PPO.load("insider_policy")

# 配置与训练时一致的环境参数
env = InsiderKyleEnv(T=10, sigma_u=0.8, sigma_v=1.2, lambda_val=0.3, max_action=3.0)

# 运行若干次模拟来收集数据（这里以1次示例，可扩展为多次取平均）
obs, _ = env.reset()
price_history = []
true_val = env.v
var_history = []
profit_per_round = []
cumulative_profit = []

done = False
while not done:
    price_history.append(env.current_price)
    var_history.append(env.current_var)
    # 使用训练策略选择动作（deterministic=True 表示选择确定性策略）
    action, _ = model.predict(obs, deterministic=True)
    # 与环境交互一步
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    profit_per_round.append(reward)
    # 记录累计收益
    cumulative_profit.append(sum(profit_per_round))

# 最后一轮结束后再记录最终价格和方差
price_history.append(env.current_price)
var_history.append(env.current_var)

# 绘制价格路径与真实价值
plt.figure(figsize=(6,4))
plt.plot(range(len(price_history)), price_history, marker='o', label='价格 $p_t$')
plt.hlines(true_val, 0, len(price_history)-1, colors='r', linestyles='--', label='真实价值 $v$')
plt.xlabel('轮次 t')
plt.ylabel('价格')
plt.title('价格路径与真实价值')
plt.legend()
plt.tight_layout()
plt.savefig('price_vs_value.png')
plt.close()

# 绘制条件方差路径
plt.figure(figsize=(6,4))
plt.plot(range(len(var_history)), var_history, marker='s', color='orange')
plt.xlabel('轮次 t')
plt.ylabel('Var[v | 信息]')
plt.title('条件方差 Var[v|p] 随轮次变化')
plt.tight_layout()
plt.savefig('variance_path.png')
plt.close()

# 绘制每轮收益和累计收益
plt.figure(figsize=(6,4))
rounds = range(1, len(profit_per_round)+1)
plt.bar(rounds, profit_per_round, alpha=0.6, label='每轮利润')
plt.plot(rounds, cumulative_profit, marker='o', color='green', label='累计利润')
plt.xlabel('轮次 t')
plt.ylabel('利润')
plt.title('内幕交易者逐轮及累计收益')
plt.legend()
plt.tight_layout()
plt.savefig('profit_path.png')
plt.close()

print("可视化图表已生成并保存为 price_vs_value.png, variance_path.png, profit_path.png")