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
    segments = [0]  # 新增：段信息，从第0段开始
    segment_boundaries = [0]  # 新增：段边界标记，第一段从0开始
    true_values_history = [env.v]  # 新增：记录每步的真实价值
    segment_true_values = [env.v]  # 新增：记录每段的真实价值，包含第一段
    
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
        true_values_history.append(env.v)  # 记录当前真实价值
        
        # 收集多段信息数据（从info中获取）
        current_segment = info.get('outer_epoch', 1) - 1  # 转换为0开始的索引
        segments.append(current_segment)
            
        # 标记段边界和收集段真实价值
        if info.get('segment_boundary', False):
            segment_boundaries.append(step)
            
        # 如果检测到段切换，记录新段的真实价值
        if info.get('segment_switch', False):
            if env.v not in segment_true_values:
                segment_true_values.append(env.v)
        
        step += 1
    
    # 确保收集到所有段的真实价值
    if hasattr(env, 'value_hist'):
        segment_true_values = env.value_hist.copy()
    
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
        'segment_boundaries': segment_boundaries,
        'true_values_history': true_values_history,
        'segment_true_values': segment_true_values
    }

def plot_basic_results(data, config_name, save_dir="./plots"):
    """绘制基础结果图表（支持多段信息可视化）"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否为多段信息
    is_multi_segment = len(set(data.get('segments', [0]))) > 1
    
    # 图1: 价格路径与真实价值
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(data['price_history'])), data['price_history'], 
             marker='o', label='价格 $p_t$', linewidth=2)
    
    # 绘制多段真实价值
    if is_multi_segment and 'segment_boundaries' in data and 'segment_true_values' in data:
        segment_boundaries = data['segment_boundaries']
        segment_true_values = data['segment_true_values']
        
        # 为每段绘制真实价值线
        for i, true_val in enumerate(segment_true_values):
            start_x = segment_boundaries[i] if i < len(segment_boundaries) else 0
            end_x = segment_boundaries[i+1] if i+1 < len(segment_boundaries) else len(data['price_history'])-1
            
            plt.hlines(true_val, start_x, end_x, 
                      colors=plt.cm.Set1(i % 9), linestyles='--', 
                      label=f'段{i+1}真实价值 $v_{i+1}$={true_val:.3f}', linewidth=2, alpha=0.8)
        
        # 添加段边界标记
        for i, boundary in enumerate(segment_boundaries):
            if boundary < len(data['price_history']):
                plt.axvline(x=boundary, color='orange', linestyle=':', alpha=0.7, 
                           label='段边界' if i == 0 else "")
    else:
        # 单段情况，显示单一真实价值
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

def plot_multi_segment_values(data, config_name, save_dir="./plots"):
    """绘制多段信息的真实价值演化图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 子图1: 价格路径与分段真实价值
    ax1.plot(range(len(data['price_history'])), data['price_history'], 
             'b-', label='价格 $p_t$', linewidth=2)
    
    # 绘制分段真实价值
    segment_boundaries = data.get('segment_boundaries', [0])
    segment_true_values = data.get('segment_true_values', [])
    
    if len(segment_true_values) > 0:
        for i, (start_idx, true_val) in enumerate(zip(segment_boundaries, segment_true_values)):
            end_idx = segment_boundaries[i+1] if i+1 < len(segment_boundaries) else len(data['price_history'])
            ax1.axhline(y=true_val, xmin=start_idx/len(data['price_history']), 
                       xmax=end_idx/len(data['price_history']), 
                       color='red', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'段{i+1}真实价值 v={true_val:.3f}' if i < 3 else '')
    
    # 标记段边界
    for boundary in segment_boundaries[1:]:
        ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7)
    
    ax1.set_xlabel('时间步 t')
    ax1.set_ylabel('价格/价值')
    ax1.set_title('价格路径与分段真实价值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 段间价值跳跃分析
    if len(segment_true_values) > 1:
        value_jumps = [segment_true_values[i+1] - segment_true_values[i] 
                      for i in range(len(segment_true_values)-1)]
        ax2.bar(range(1, len(value_jumps)+1), value_jumps, 
                color=['green' if jump > 0 else 'red' for jump in value_jumps],
                alpha=0.7)
        ax2.set_xlabel('段间跳跃')
        ax2.set_ylabel('价值变化')
        ax2.set_title('段间真实价值跳跃')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, jump in enumerate(value_jumps):
            ax2.text(i+1, jump, f'{jump:.3f}', ha='center', 
                    va='bottom' if jump > 0 else 'top')
    else:
        ax2.text(0.5, 0.5, '单段信息\n无跳跃分析', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('段间真实价值跳跃')
    
    # 子图3: 方差演化（如果有数据）
    if 'var_history' in data and len(data['var_history']) > 0:
        ax3.plot(range(len(data['var_history'])), data['var_history'], 
                 'g-', label='方差 $\sigma^2_t$', linewidth=2)
        ax3.set_xlabel('时间步 t')
        ax3.set_ylabel('方差')
        ax3.set_title('方差演化路径')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, '方差数据\n不可用', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('方差演化路径')
    
    # 子图4: 价格与真实价值的关系统计
    if len(segment_true_values) > 0:
        # 计算每段的平均价格
        segment_avg_prices = []
        for i, start_idx in enumerate(segment_boundaries):
            end_idx = segment_boundaries[i+1] if i+1 < len(segment_boundaries) else len(data['price_history'])
            segment_prices = data['price_history'][start_idx:end_idx]
            if len(segment_prices) > 0:
                segment_avg_prices.append(np.mean(segment_prices))
        
        if len(segment_avg_prices) == len(segment_true_values):
            ax4.scatter(segment_true_values, segment_avg_prices, 
                       s=100, alpha=0.7, c=range(len(segment_true_values)), cmap='viridis')
            
            # 添加对角线参考
            min_val = min(min(segment_true_values), min(segment_avg_prices))
            max_val = max(max(segment_true_values), max(segment_avg_prices))
            ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='完美定价线')
            
            ax4.set_xlabel('真实价值 v')
            ax4.set_ylabel('平均价格 $\\bar{p}$')
            ax4.set_title('各段真实价值vs平均价格')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 添加段标签
            for i, (true_val, avg_price) in enumerate(zip(segment_true_values, segment_avg_prices)):
                ax4.annotate(f'段{i+1}', (true_val, avg_price), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        else:
            ax4.text(0.5, 0.5, '价格数据\n不完整', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('各段真实价值vs平均价格')
    else:
        ax4.text(0.5, 0.5, '真实价值数据\n不可用', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('各段真实价值vs平均价格')
    
    plt.tight_layout()
    
    # 保存图片
    import os
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{config_name}_multi_segment_values.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"多段真实价值分析图已保存: {filepath}")
    


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
        
        # 绘制多段真实价值分析图
        plot_multi_segment_values(data, config_name)
        
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
        
        # 绘制多段真实价值分析图（如果适用）
        if 'segments' in data and len(set(data['segments'])) > 1:
            print(f"检测到多段信息配置，绘制多段真实价值分析图...")
            plot_multi_segment_values(data, config_name, save_dir)
        
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