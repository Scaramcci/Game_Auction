import matplotlib.pyplot as plt
import numpy as np
import os
from stable_baselines3 import PPO
from env import EnhancedInsiderKyleEnv
import font_config
from multi_segment_analysis import plot_multi_segment_analysis, generate_multi_segment_report

# 设置中文字体支持
font_config.setup_chinese_font()

def load_model_and_env(config_name, models_dir="./models"):
    """加载训练好的模型和对应环境配置"""
    model_path = os.path.join(models_dir, f"{config_name}_policy")
    
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"模型文件不存在: {model_path}.zip")
    
    model = PPO.load(model_path)
    
    # 根据配置名称设置环境参数
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

def run_single_episode(model, env, deterministic=True):
    """运行单个episode并收集数据（支持多段信息）"""
    obs, _ = env.reset()
    
    price_history = [env.current_price]
    true_val = env.v
    var_history = [env.cur_var]
    profit_per_round = []
    cumulative_profit = []
    segments = []  # 新增：段信息
    segment_boundaries = []  # 新增：段边界标记
    
    done = False
    step = 0
    while not done:
        # 使用训练策略选择动作
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # 与环境交互一步
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 记录数据
        price_history.append(env.current_price)
        var_history.append(env.cur_var)
        profit_per_round.append(reward)
        cumulative_profit.append(sum(profit_per_round))
        
        # 收集多段信息数据
        if hasattr(env, 'outer_epoch'):
            segments.append(env.outer_epoch)
        else:
            segments.append(0)
            
        # 标记段边界
        if hasattr(env, 'is_segment_boundary') and env.is_segment_boundary():
            segment_boundaries.append(step)
        
        step += 1
    
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
        'market_depth': env.get_market_depth(),
        'segments': segments,
        'segment_boundaries': segment_boundaries
    }

def plot_basic_results(data, config_name, save_dir="./plots"):
    """绘制基础结果图表（支持多段信息可视化）"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否为多段信息
    is_multi_segment = len(set(data.get('segments', [0]))) > 1
    
    # 图1: 价格路径与真实价值
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data['price_history'])), data['price_history'], 
             marker='o', label='价格 $p_t$', linewidth=2)
    plt.hlines(data['true_val'], 0, len(data['price_history'])-1, 
               colors='r', linestyles='--', label='真实价值 $v$', linewidth=2)
    
    # 添加段边界标记
    if is_multi_segment and 'segment_boundaries' in data:
        for i, boundary in enumerate(data['segment_boundaries']):
            if boundary < len(data['price_history']):
                plt.axvline(x=boundary, color='orange', linestyle=':', alpha=0.7, 
                           label='段边界' if i == 0 else "")
    
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
    
    # 添加段边界标记
    if is_multi_segment and 'segment_boundaries' in data:
        for boundary in data['segment_boundaries']:
            if boundary < len(data['var_history']):
                plt.axvline(x=boundary, color='orange', linestyle=':', alpha=0.7)
    
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
    
    # 添加段边界标记
    if is_multi_segment and 'segment_boundaries' in data:
        for boundary in data['segment_boundaries']:
            if boundary < len(rounds):
                plt.axvline(x=boundary+1, color='orange', linestyle=':', alpha=0.7)
    
    plt.xlabel('轮次 t')
    plt.ylabel('利润')
    plt.title(f'内幕交易者逐轮及累计收益 - {config_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加多段信息说明
    if is_multi_segment:
        plt.figtext(0.02, 0.02, f'多段信息配置 - 段数: {len(set(data.get("segments", [0])))}', 
                   fontsize=10, style='italic', alpha=0.7)
    
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

def analyze_configuration(config_name):
    """分析单个配置（支持多段信息分析）"""
    print(f"\n分析配置: {config_name}")
    
    # 加载模型和环境
    model, env = load_model_and_env(config_name)
    
    # 运行episode收集数据
    data = run_single_episode(model, env)
    
    # 绘制基础结果
    plot_basic_results(data, config_name)
    
    # 检查是否为多段信息配置
    if 'segments' in data and len(set(data['segments'])) > 1:
        print(f"检测到多段信息配置，进行专门分析...")
        
        # 绘制多段信息分析图
        plot_multi_segment_analysis(data, config_name)
        
        # 生成多段信息报告
        multi_segment_report = generate_multi_segment_report(data, config_name)
        
        return data, multi_segment_report
    
    print(f"配置 {config_name} 分析完成")
    return data, None

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
    """主可视化函数（支持多段信息分析）"""
    configs = ['baseline_static', 'baseline_dynamic', 'high_noise', 'low_noise', 'long_term', 'multi_segment_3', 'multi_segment_5']
    
    print("开始可视化所有配置...")
    print("V3版本新增多段信息配置专门分析")
    
    multi_segment_results = {}
    
    for config in configs:
        try:
            data, multi_segment_report = analyze_configuration(config)
            if multi_segment_report:
                multi_segment_results[config] = multi_segment_report
        except Exception as e:
            print(f"配置 {config} 可视化失败: {e}")
            continue
    
    # 生成多段信息配置对比
    if multi_segment_results:
        print(f"\n{'='*60}")
        print("多段信息配置对比总结")
        print(f"{'='*60}")
        
        for config, report in multi_segment_results.items():
            print(f"\n{config}:")
            print(f"  段数: {report['num_segments']}")
            print(f"  整体信息效率: {report['overall_efficiency']:.4f}")
            print(f"  整体价格准确性: {report['overall_accuracy']:.4f}")
            if report['transition_analysis']:
                print(f"  价格连续性得分: {report['transition_analysis']['price_continuity_score']:.4f}")
    
    print("\n所有配置可视化完成!")
    print("结果保存在 ./plots/ 目录中")
    print("多段信息配置包含额外的专门分析图表")

if __name__ == "__main__":
    results = main()