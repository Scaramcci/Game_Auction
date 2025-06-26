import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from stable_baselines3 import PPO
from env import EnhancedInsiderKyleEnv
import os
from font_config import setup_chinese_font

# 设置中文字体支持
setup_chinese_font()

def load_model_and_env(config_name, models_dir="./models"):
    """加载训练好的模型和对应环境配置"""
    model_path = os.path.join(models_dir, f"{config_name}_policy")
    
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"模型文件不存在: {model_path}.zip")
    
    model = PPO.load(model_path)
    
    # 根据配置名称设置环境参数
    env_configs = {
        'baseline_static': {'T': 10, 'sigma_u': 0.8, 'sigma_v': 1.2, 'lambda_val': 0.3, 'max_action': 3.0, 'dynamic_lambda': False},
        'baseline_dynamic': {'T': 10, 'sigma_u': 0.8, 'sigma_v': 1.2, 'lambda_val': 0.3, 'max_action': 3.0, 'dynamic_lambda': True},
        'high_noise': {'T': 10, 'sigma_u': 1.5, 'sigma_v': 1.2, 'lambda_val': 0.2, 'max_action': 3.0, 'dynamic_lambda': False},
        'low_noise': {'T': 10, 'sigma_u': 0.5, 'sigma_v': 1.2, 'lambda_val': 0.4, 'max_action': 3.0, 'dynamic_lambda': False},
        'long_term': {'T': 20, 'sigma_u': 0.8, 'sigma_v': 1.5, 'lambda_val': 0.25, 'max_action': 2.5, 'dynamic_lambda': True}
    }
    
    if config_name not in env_configs:
        raise ValueError(f"未知的配置名称: {config_name}")
    
    env = EnhancedInsiderKyleEnv(**env_configs[config_name])
    return model, env

def run_single_episode(model, env, deterministic=True):
    """运行单个episode并收集数据"""
    obs, _ = env.reset()
    
    price_history = [env.current_price]
    true_val = env.v
    var_history = [env.current_var]
    profit_per_round = []
    cumulative_profit = []
    
    done = False
    while not done:
        # 使用训练策略选择动作
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # 与环境交互一步
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 记录数据
        price_history.append(env.current_price)
        var_history.append(env.current_var)
        profit_per_round.append(reward)
        cumulative_profit.append(sum(profit_per_round))
    
    return {
        'price_history': price_history,
        'true_val': true_val,
        'var_history': var_history,
        'profit_per_round': profit_per_round,
        'cumulative_profit': cumulative_profit,
        'lambda_hist': env.lambda_hist,
        'beta_hist': env.beta_hist,
        'price_impact_hist': env.price_impact_hist,
        'order_flow_hist': env.order_flow_hist,
        'noise_hist': env.noise_hist,
        'action_hist': env.action_hist,
        'market_depth': env.get_market_depth()
    }

def plot_basic_results(data, config_name, save_dir="./plots"):
    """绘制基础结果图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 图1: 价格路径与真实价值
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data['price_history'])), data['price_history'], 
             marker='o', label='价格 $p_t$', linewidth=2)
    plt.hlines(data['true_val'], 0, len(data['price_history'])-1, 
               colors='r', linestyles='--', label='真实价值 $v$', linewidth=2)
    plt.xlabel('轮次 t')
    plt.ylabel('价格')
    plt.title(f'价格路径与真实价值 - {config_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{config_name}_price_vs_value.png'), dpi=300)
    plt.close()
    
    # 图2: 条件方差路径
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data['var_history'])), data['var_history'], 
             marker='s', color='orange', linewidth=2)
    plt.xlabel('轮次 t')
    plt.ylabel('Var[v | 信息]')
    plt.title(f'条件方差 Var[v|p] 随轮次变化 - {config_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{config_name}_variance_path.png'), dpi=300)
    plt.close()
    
    # 图3: 每轮收益和累计收益
    plt.figure(figsize=(10, 6))
    rounds = range(1, len(data['profit_per_round'])+1)
    plt.bar(rounds, data['profit_per_round'], alpha=0.6, label='每轮利润')
    plt.plot(rounds, data['cumulative_profit'], marker='o', 
             color='green', label='累计利润', linewidth=2)
    plt.xlabel('轮次 t')
    plt.ylabel('利润')
    plt.title(f'内幕交易者逐轮及累计收益 - {config_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{config_name}_profit_path.png'), dpi=300)
    plt.close()

def plot_kyle_metrics(data, config_name, save_dir="./plots"):
    """绘制Kyle模型特有指标"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 图4: λₜ (价格冲击系数) 随时间变化
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(data['lambda_hist'])+1), data['lambda_hist'], 
             marker='o', color='blue', linewidth=2, label='实际 $\\lambda_t$')
    plt.xlabel('轮次 t')
    plt.ylabel('价格冲击系数 $\\lambda_t$')
    plt.title(f'价格冲击系数 $\\lambda_t$ 随轮次变化 - {config_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{config_name}_lambda_evolution.png'), dpi=300)
    plt.close()
    
    # 图5: βₜ (交易强度) 随时间变化
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(data['beta_hist'])+1), data['beta_hist'], 
             marker='s', color='red', linewidth=2, label='实际 $\\beta_t$')
    plt.xlabel('轮次 t')
    plt.ylabel('交易强度 $\\beta_t$')
    plt.title(f'交易强度 $\\beta_t$ 随轮次变化 - {config_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{config_name}_beta_evolution.png'), dpi=300)
    plt.close()
    
    # 图6: 市场深度 (1/λₜ) 随时间变化
    if len(data['market_depth']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(data['market_depth'])+1), data['market_depth'], 
                 marker='^', color='purple', linewidth=2, label='市场深度 $1/\\lambda_t$')
        plt.xlabel('轮次 t')
        plt.ylabel('市场深度 $1/\\lambda_t$')
        plt.title(f'市场深度随轮次变化 - {config_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{config_name}_market_depth.png'), dpi=300)
        plt.close()

def visualize_config(config_name, models_dir="./models", save_dir="./plots"):
    """可视化单个配置的结果"""
    print(f"\n正在可视化配置: {config_name}")
    
    try:
        # 加载模型和环境
        model, env = load_model_and_env(config_name, models_dir)
        
        # 运行episode收集数据
        data = run_single_episode(model, env)
        
        # 绘制基础图表
        plot_basic_results(data, config_name, save_dir)
        
        # 绘制Kyle指标
        plot_kyle_metrics(data, config_name, save_dir)
        
        print(f"配置 {config_name} 的图表已保存到 {save_dir} 目录")
        
        return data
        
    except Exception as e:
        print(f"可视化配置 {config_name} 时出错: {e}")
        return None

def main():
    """主可视化函数"""
    configs = ['baseline_static', 'baseline_dynamic', 'high_noise', 'low_noise', 'long_term']
    
    print("开始可视化所有配置...")
    
    all_data = {}
    for config in configs:
        data = visualize_config(config)
        if data is not None:
            all_data[config] = data
    
    print(f"\n可视化完成！共处理了 {len(all_data)} 个配置")
    print("图表保存在 ./plots/ 目录中")
    
    return all_data

if __name__ == "__main__":
    results = main()