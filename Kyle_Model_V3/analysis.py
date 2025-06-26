import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from stable_baselines3 import PPO
from env import EnhancedInsiderKyleEnv
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from font_config import setup_chinese_font

# 设置中文字体支持
setup_chinese_font()

def load_model_and_env(config_name, models_dir="./models"):
    """加载训练好的模型和对应环境配置"""
    model_path = os.path.join(models_dir, f"{config_name}_policy")
    
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"模型文件不存在: {model_path}.zip")
    
    model = PPO.load(model_path)
    
    # 环境配置
    env_configs = {
        'baseline_static': {'T': 10, 'sigma_u': 0.8, 'sigma_v': 1.2, 'lambda_val': 0.3, 'max_action': 3.0, 'dynamic_lambda': False, 'super_horizon': 1},
        'baseline_dynamic': {'T': 10, 'sigma_u': 0.8, 'sigma_v': 1.2, 'lambda_val': 0.3, 'max_action': 3.0, 'dynamic_lambda': True, 'super_horizon': 1},
        'high_noise': {'T': 10, 'sigma_u': 1.5, 'sigma_v': 1.2, 'lambda_val': 0.2, 'max_action': 3.0, 'dynamic_lambda': True, 'super_horizon': 1},
        'low_noise': {'T': 10, 'sigma_u': 0.5, 'sigma_v': 1.2, 'lambda_val': 0.4, 'max_action': 3.0, 'dynamic_lambda': True, 'super_horizon': 1},
        'long_term': {'T': 20, 'sigma_u': 0.8, 'sigma_v': 1.5, 'lambda_val': 0.25, 'max_action': 2.5, 'dynamic_lambda': True, 'super_horizon': 1},
        'multi_segment_3': {'T': 10, 'sigma_u': 0.8, 'sigma_v': 1.2, 'lambda_val': 0.3, 'max_action': 3.0, 'dynamic_lambda': True, 'super_horizon': 3},
        'multi_segment_5': {'T': 10, 'sigma_u': 0.8, 'sigma_v': 1.2, 'lambda_val': 0.3, 'max_action': 3.0, 'dynamic_lambda': True, 'super_horizon': 5}
    }
    
    if config_name not in env_configs:
        raise ValueError(f"未知的配置名称: {config_name}")
    
    env = EnhancedInsiderKyleEnv(**env_configs[config_name])
    return model, env

def collect_batch_data(model, env, episodes=1000, deterministic=True):
    """批量采样收集数据用于回归分析（支持多段信息）"""
    print(f"正在收集 {episodes} 个episode的数据...")
    
    all_data = []
    episode_summaries = []
    
    for episode in range(episodes):
        if (episode + 1) % 100 == 0:
            print(f"已完成 {episode + 1}/{episodes} episodes")
        
        obs, _ = env.reset()
        episode_data = []
        total_profit = 0
        
        done = False
        step = 0
        while not done:
            prev_price = env.current_price
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录每步数据
            step_data = {
                'episode': episode,
                'step': step,
                'dP': env.current_price - prev_price,  # 价格变化
                'Q': info['order_flow'],  # 总订单流
                'x': env.action_hist[-1],  # 内幕交易量
                'u': info['noise'],  # 噪声交易
                'lambda_t': info['lambda_t'],  # 当前λ
                'beta_t': info['beta_t'],  # 当前β
                'price': env.current_price,
                'true_val': env.v,
                'reward': reward,
                'var_t': info['Var(v|info)'],
                'segment': info.get('outer_epoch', 0),  # 新增：段信息
                'segment_boundary': info.get('segment_boundary', False)  # 新增：段边界
            }
            episode_data.append(step_data)
            total_profit += reward
            step += 1
        
        # 记录episode汇总
        episode_summary = {
            'episode': episode,
            'total_profit': total_profit,
            'final_price': env.current_price,
            'true_val': env.v,
            'price_error': abs(env.current_price - env.v),
            'final_var': env.current_var,
            'info_incorporation': 1 - (env.current_var / env.sigma_v**2)
        }
        episode_summaries.append(episode_summary)
        all_data.extend(episode_data)
    
    return pd.DataFrame(all_data), pd.DataFrame(episode_summaries)

def estimate_price_impact_regression(df):
    """使用横截面回归估计价格冲击系数"""
    # 过滤掉Q=0的情况避免除零
    valid_data = df[np.abs(df['Q']) > 1e-8].copy()
    
    if len(valid_data) == 0:
        return None, None, None
    
    # 回归: ΔP = λ * Q + ε
    X = valid_data[['Q']]
    y = valid_data['dP']
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    lambda_reg = reg.coef_[0]
    r2 = r2_score(y, reg.predict(X))
    
    # 计算标准误
    residuals = y - reg.predict(X)
    mse = np.mean(residuals**2)
    var_Q = np.var(valid_data['Q'])
    se_lambda = np.sqrt(mse / (len(valid_data) * var_Q))
    
    return lambda_reg, se_lambda, r2

def calculate_theoretical_values(env_params):
    """计算理论值"""
    sigma_u = env_params.get('sigma_u', 0.8)
    sigma_v = env_params.get('sigma_v', 1.2)
    
    # 理论β (单期近似)
    beta_theory = sigma_u / sigma_v
    
    # 理论λ
    lambda_theory = 1.0 / (2.0 * beta_theory)
    
    return {
        'beta_theory': beta_theory,
        'lambda_theory': lambda_theory,
        'market_depth_theory': 1.0 / lambda_theory
    }

def analyze_price_efficiency(episode_df):
    """分析价格效率"""
    # 价格-真值回归 R²
    if len(episode_df) > 1:
        X = episode_df[['final_price']]
        y = episode_df['true_val']
        reg = LinearRegression()
        reg.fit(X, y)
        price_efficiency_r2 = r2_score(y, reg.predict(X))
    else:
        price_efficiency_r2 = 0
    
    # 平均价格误差
    mean_price_error = episode_df['price_error'].mean()
    
    # 信息融入比例
    mean_info_incorporation = episode_df['info_incorporation'].mean()
    
    return {
        'price_efficiency_r2': price_efficiency_r2,
        'mean_price_error': mean_price_error,
        'mean_info_incorporation': mean_info_incorporation
    }

def analyze_multi_segment_performance(step_df):
    """分析多段信息配置的特定性能指标"""
    if 'segment' not in step_df.columns:
        return None
    
    segments = step_df['segment'].unique()
    if len(segments) <= 1:
        return None
    
    # 计算段间价格连续性
    segment_boundaries = step_df[step_df['segment_boundary'] == True]
    if len(segment_boundaries) > 0:
        price_jumps = np.abs(segment_boundaries['dP'])
        price_continuity = 1.0 / (1.0 + np.mean(price_jumps))
    else:
        price_continuity = 1.0
    
    # 计算信息传递效率
    segment_efficiencies = []
    for seg in segments:
        seg_data = step_df[step_df['segment'] == seg]
        if len(seg_data) > 1:
            final_var = seg_data['var_t'].iloc[-1]
            initial_var = seg_data['var_t'].iloc[0]
            if initial_var > 0:
                efficiency = (initial_var - final_var) / initial_var
                segment_efficiencies.append(max(0, efficiency))
    
    info_transfer_efficiency = np.mean(segment_efficiencies) if segment_efficiencies else 0
    
    return {
        'price_continuity': price_continuity,
        'info_transfer_efficiency': info_transfer_efficiency,
        'num_segments': len(segments),
        'segment_efficiencies': segment_efficiencies
    }

def generate_analysis_report(config_name, step_df, episode_df, env_params):
    """生成分析报告（支持多段信息分析）"""
    print(f"\n=== {config_name} 配置分析报告 ===")
    
    # 检查是否为多段信息配置
    is_multi_segment = env_params.get('super_horizon', 1) > 1
    
    # 1. 价格冲击回归
    lambda_reg, se_lambda, r2_lambda = estimate_price_impact_regression(step_df)
    
    # 2. 理论值计算
    theoretical = calculate_theoretical_values(env_params)
    
    # 3. 价格效率分析
    efficiency = analyze_price_efficiency(episode_df)
    
    # 4. 基础统计
    mean_beta = step_df['beta_t'].mean()
    mean_lambda = step_df['lambda_t'].mean()
    mean_profit = episode_df['total_profit'].mean()
    std_profit = episode_df['total_profit'].std()
    
    # 5. 多段信息分析
    segment_analysis = None
    if is_multi_segment:
        segment_analysis = analyze_multi_segment_performance(step_df)
        print(f"\n【多段信息配置】")
        print(f"  配置段数: {env_params['super_horizon']}")
        if segment_analysis:
            print(f"  实际段数: {segment_analysis['num_segments']}")
            print(f"  段间价格连续性: {segment_analysis['price_continuity']:.4f}")
            print(f"  信息传递效率: {segment_analysis['info_transfer_efficiency']:.4f}")
    
    # 打印报告
    print(f"\n【价格冲击分析】")
    if lambda_reg is not None:
        print(f"  回归估计 λ: {lambda_reg:.4f} (标准误: {se_lambda:.4f})")
        print(f"  回归 R²: {r2_lambda:.4f}")
        print(f"  理论 λ: {theoretical['lambda_theory']:.4f}")
        print(f"  估计偏差: {abs(lambda_reg - theoretical['lambda_theory']):.4f}")
    else:
        print(f"  回归失败 (数据不足)")
    
    print(f"\n【交易强度分析】")
    print(f"  平均 β: {mean_beta:.4f}")
    print(f"  理论 β: {theoretical['beta_theory']:.4f}")
    print(f"  估计偏差: {abs(mean_beta - theoretical['beta_theory']):.4f}")
    
    print(f"\n【市场深度分析】")
    if mean_lambda > 0:
        empirical_depth = 1.0 / mean_lambda
        print(f"  经验市场深度: {empirical_depth:.4f}")
        print(f"  理论市场深度: {theoretical['market_depth_theory']:.4f}")
    
    print(f"\n【价格效率分析】")
    print(f"  价格-真值 R²: {efficiency['price_efficiency_r2']:.4f}")
    print(f"  平均价格误差: {efficiency['mean_price_error']:.4f}")
    print(f"  平均信息融入率: {efficiency['mean_info_incorporation']:.4f}")
    
    print(f"\n【收益分析】")
    print(f"  平均总收益: {mean_profit:.4f}")
    print(f"  收益标准差: {std_profit:.4f}")
    print(f"  夏普比率: {mean_profit/max(std_profit, 1e-8):.4f}")
    
    # 返回结果字典
    return {
        'config_name': config_name,
        'lambda_reg': lambda_reg,
        'se_lambda': se_lambda,
        'r2_lambda': r2_lambda,
        'mean_beta': mean_beta,
        'mean_lambda': mean_lambda,
        'mean_profit': mean_profit,
        'std_profit': std_profit,
        'theoretical': theoretical,
        'efficiency': efficiency,
        'segment_analysis': segment_analysis
    }

def plot_regression_analysis(step_df, config_name, save_dir="./analysis_plots"):
    """绘制回归分析图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 价格冲击回归散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(step_df['Q'], step_df['dP'], alpha=0.5, s=10)
    
    # 拟合回归线
    valid_data = step_df[np.abs(step_df['Q']) > 1e-8]
    if len(valid_data) > 0:
        X = valid_data[['Q']]
        y = valid_data['dP']
        reg = LinearRegression()
        reg.fit(X, y)
        
        Q_range = np.linspace(valid_data['Q'].min(), valid_data['Q'].max(), 100)
        dP_pred = reg.predict(Q_range.reshape(-1, 1))
        plt.plot(Q_range, dP_pred, 'r-', linewidth=2, 
                label=f'回归线: ΔP = {reg.coef_[0]:.3f} × Q')
    
    plt.xlabel('订单流 Q')
    plt.ylabel('价格变化 ΔP')
    plt.title(f'价格冲击回归分析 - {config_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{config_name}_price_impact_regression.png'), dpi=300)
    plt.close()

def compare_configurations(all_results, save_dir="./analysis_plots"):
    """比较不同配置的结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    configs = list(all_results.keys())
    
    # 提取比较数据
    lambda_values = [all_results[c]['lambda_reg'] if all_results[c]['lambda_reg'] is not None else 0 for c in configs]
    beta_values = [all_results[c]['mean_beta'] for c in configs]
    profit_values = [all_results[c]['mean_profit'] for c in configs]
    r2_values = [all_results[c]['efficiency']['price_efficiency_r2'] for c in configs]
    
    # 创建比较图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # λ比较
    ax1.bar(configs, lambda_values)
    ax1.set_title('价格冲击系数 λ 比较')
    ax1.set_ylabel('λ 值')
    ax1.tick_params(axis='x', rotation=45)
    
    # β比较
    ax2.bar(configs, beta_values)
    ax2.set_title('交易强度 β 比较')
    ax2.set_ylabel('β 值')
    ax2.tick_params(axis='x', rotation=45)
    
    # 收益比较
    ax3.bar(configs, profit_values)
    ax3.set_title('平均总收益比较')
    ax3.set_ylabel('收益')
    ax3.tick_params(axis='x', rotation=45)
    
    # 价格效率比较
    ax4.bar(configs, r2_values)
    ax4.set_title('价格效率 R² 比较')
    ax4.set_ylabel('R² 值')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'configuration_comparison.png'), dpi=300)
    plt.close()

def analyze_configuration(config_name, episodes=1000, models_dir="./models"):
    """分析单个配置"""
    print(f"\n开始分析配置: {config_name}")
    
    try:
        # 加载模型和环境
        model, env = load_model_and_env(config_name, models_dir)
        
        # 收集批量数据
        step_df, episode_df = collect_batch_data(model, env, episodes)
        
        # 生成分析报告
        env_configs = {
            'baseline_static': {'sigma_u': 0.8, 'sigma_v': 1.2, 'super_horizon': 1},
            'baseline_dynamic': {'sigma_u': 0.8, 'sigma_v': 1.2, 'super_horizon': 1},
            'high_noise': {'sigma_u': 1.5, 'sigma_v': 1.2, 'super_horizon': 1},
            'low_noise': {'sigma_u': 0.5, 'sigma_v': 1.2, 'super_horizon': 1},
            'long_term': {'sigma_u': 0.8, 'sigma_v': 1.5, 'super_horizon': 1},
            'multi_segment_3': {'sigma_u': 0.8, 'sigma_v': 1.2, 'super_horizon': 3},
            'multi_segment_5': {'sigma_u': 0.8, 'sigma_v': 1.2, 'super_horizon': 5}
        }
        
        results = generate_analysis_report(config_name, step_df, episode_df, env_configs[config_name])
        
        # 绘制回归分析图
        plot_regression_analysis(step_df, config_name)
        
        return results, step_df, episode_df
        
    except Exception as e:
        print(f"分析配置 {config_name} 时出错: {e}")
        return None, None, None

def main():
    """主分析函数"""
    configs = ['baseline_static', 'baseline_dynamic', 'high_noise', 'low_noise', 'long_term', 'multi_segment_3', 'multi_segment_5']
    
    print("开始批量分析所有配置...")
    print("注意: 每个配置将运行1000个episodes，可能需要较长时间")
    
    all_results = {}
    
    for config in configs:
        results, step_df, episode_df = analyze_configuration(config, episodes=1000)
        if results is not None:
            all_results[config] = results
    
    # 生成配置比较图
    if len(all_results) > 1:
        compare_configurations(all_results)
        print("\n配置比较图已生成: ./analysis_plots/configuration_comparison.png")
    
    print(f"\n=== 分析完成 ===")
    print(f"共分析了 {len(all_results)} 个配置")
    print("详细结果已保存在 ./analysis_plots/ 目录中")
    
    return all_results

if __name__ == "__main__":
    results = main()