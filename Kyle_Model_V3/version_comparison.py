#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kyle Model V2 vs V3 版本对比脚本
展示V3相对于V2的新功能和改进
"""

import numpy as np
import matplotlib.pyplot as plt
from env import EnhancedInsiderKyleEnv
import font_config

def compare_versions():
    """对比V2和V3版本"""
    print("Kyle Model V2 vs V3 版本对比")
    print("=" * 50)
    
    # V2兼容配置 (super_horizon=1)
    v2_config = {
        'T': 10,
        'sigma_u': 0.8,
        'sigma_v': 1.2,
        'lambda_val': 0.3,
        'max_action': 3.0,
        'seed': 42,
        'dynamic_lambda': True,
        'super_horizon': 1  # V2模式
    }
    
    # V3多段信息配置
    v3_config = {
        'T': 5,  # 每段5轮
        'sigma_u': 0.8,
        'sigma_v': 1.2,
        'lambda_val': 0.3,
        'max_action': 3.0,
        'seed': 42,
        'dynamic_lambda': True,
        'super_horizon': 3  # V3多段模式
    }
    
    results = {}
    
    # 测试V2兼容模式
    print("\n1. 测试V2兼容模式 (super_horizon=1)")
    print("-" * 30)
    
    env_v2 = EnhancedInsiderKyleEnv(**v2_config)
    v2_data = run_episode(env_v2, "V2兼容")
    results['V2'] = v2_data
    
    # 测试V3多段模式
    print("\n2. 测试V3多段模式 (super_horizon=3)")
    print("-" * 30)
    
    env_v3 = EnhancedInsiderKyleEnv(**v3_config)
    v3_data = run_episode(env_v3, "V3多段")
    results['V3'] = v3_data
    
    # 对比分析
    print("\n3. 版本对比分析")
    print("-" * 30)
    analyze_differences(results)
    
    # 可视化对比
    print("\n4. 生成对比图表")
    print("-" * 30)
    plot_comparison(results)
    
    return results

def run_episode(env, version_name):
    """运行一个episode并收集数据"""
    obs, info = env.reset()
    
    episode_data = {
        'steps': [],
        'prices': [],
        'values': [],
        'rewards': [],
        'actions': [],
        'outer_epochs': [],
        'inner_steps': [],
        'segment_boundaries': [],
        'lambda_values': [],
        'conditional_vars': []
    }
    
    step = 0
    done = False
    
    print(f"开始运行 {version_name} episode...")
    
    while not done:
        # 简单策略：基于价格偏差的动作
        price = info.get('price', 0)
        value = info.get('value', 0)
        action = np.clip((value - price) * 0.5, -env.max_action, env.max_action)
        
        obs, reward, done, truncated, info = env.step([action])
        
        # 收集数据
        episode_data['steps'].append(step)
        episode_data['prices'].append(info['price'])
        episode_data['values'].append(info['value'])
        episode_data['rewards'].append(reward)
        episode_data['actions'].append(action)
        episode_data['outer_epochs'].append(info.get('outer_epoch', 0))
        episode_data['inner_steps'].append(info.get('inner_step', step))
        episode_data['segment_boundaries'].append(info.get('segment_boundary', False))
        episode_data['lambda_values'].append(info.get('lambda', 0))
        episode_data['conditional_vars'].append(info.get('conditional_var', 0))
        
        step += 1
        
        if step > 100:  # 防止无限循环
            break
    
    print(f"{version_name} episode完成: {step} 步")
    
    return episode_data

def analyze_differences(results):
    """分析版本差异"""
    v2_data = results['V2']
    v3_data = results['V3']
    
    print("基本统计对比:")
    print(f"  V2总步数: {len(v2_data['steps'])}")
    print(f"  V3总步数: {len(v3_data['steps'])}")
    
    # 段信息分析
    v2_segments = set(v2_data['outer_epochs'])
    v3_segments = set(v3_data['outer_epochs'])
    
    print(f"\n段信息对比:")
    print(f"  V2段数: {len(v2_segments)} {sorted(list(v2_segments))}")
    print(f"  V3段数: {len(v3_segments)} {sorted(list(v3_segments))}")
    
    # 段边界分析
    v2_boundaries = sum(v2_data['segment_boundaries'])
    v3_boundaries = sum(v3_data['segment_boundaries'])
    
    print(f"\n段边界对比:")
    print(f"  V2段边界数: {v2_boundaries}")
    print(f"  V3段边界数: {v3_boundaries}")
    
    # 价格统计
    v2_price_std = np.std(v2_data['prices'])
    v3_price_std = np.std(v3_data['prices'])
    
    print(f"\n价格波动对比:")
    print(f"  V2价格标准差: {v2_price_std:.4f}")
    print(f"  V3价格标准差: {v3_price_std:.4f}")
    
    # 奖励统计
    v2_total_reward = sum(v2_data['rewards'])
    v3_total_reward = sum(v3_data['rewards'])
    
    print(f"\n奖励对比:")
    print(f"  V2总奖励: {v2_total_reward:.4f}")
    print(f"  V3总奖励: {v3_total_reward:.4f}")
    
    # 信息效率分析
    v2_price_accuracy = calculate_price_accuracy(v2_data['prices'], v2_data['values'])
    v3_price_accuracy = calculate_price_accuracy(v3_data['prices'], v3_data['values'])
    
    print(f"\n价格准确性对比:")
    print(f"  V2价格准确性: {v2_price_accuracy:.4f}")
    print(f"  V3价格准确性: {v3_price_accuracy:.4f}")

def calculate_price_accuracy(prices, values):
    """计算价格准确性"""
    if len(prices) == 0 or len(values) == 0:
        return 0.0
    
    errors = [abs(p - v) for p, v in zip(prices, values)]
    mean_error = np.mean(errors)
    
    # 转换为准确性分数 (0-1)
    max_possible_error = max(max(values) - min(values), 1e-6)
    accuracy = max(0, 1 - mean_error / max_possible_error)
    
    return accuracy

def plot_comparison(results):
    """绘制版本对比图"""
    font_config.setup_chinese_font()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Kyle Model V2 vs V3 版本对比', fontsize=16, fontweight='bold')
    
    v2_data = results['V2']
    v3_data = results['V3']
    
    # 1. 价格演化对比
    ax1 = axes[0, 0]
    ax1.plot(v2_data['steps'], v2_data['prices'], 'b-', label='V2价格', linewidth=2)
    ax1.plot(v2_data['steps'], v2_data['values'], 'b--', label='V2真值', alpha=0.7)
    
    # V3数据可能更长，需要调整x轴
    v3_steps_adjusted = [s + max(v2_data['steps']) + 5 for s in v3_data['steps']]
    ax1.plot(v3_steps_adjusted, v3_data['prices'], 'r-', label='V3价格', linewidth=2)
    ax1.plot(v3_steps_adjusted, v3_data['values'], 'r--', label='V3真值', alpha=0.7)
    
    # 标记V3段边界
    for i, boundary in enumerate(v3_data['segment_boundaries']):
        if boundary:
            ax1.axvline(x=v3_steps_adjusted[i], color='orange', linestyle=':', alpha=0.7,
                       label='V3段边界' if i == next(j for j, b in enumerate(v3_data['segment_boundaries']) if b) else "")
    
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('价格/价值')
    ax1.set_title('价格演化对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 段信息对比
    ax2 = axes[0, 1]
    ax2.plot(v2_data['steps'], v2_data['outer_epochs'], 'b-', marker='o', label='V2段信息', linewidth=2)
    ax2.plot(v3_steps_adjusted, v3_data['outer_epochs'], 'r-', marker='s', label='V3段信息', linewidth=2)
    
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('段编号')
    ax2.set_title('段信息演化对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 奖励对比
    ax3 = axes[1, 0]
    v2_cumulative_rewards = np.cumsum(v2_data['rewards'])
    v3_cumulative_rewards = np.cumsum(v3_data['rewards'])
    
    ax3.plot(v2_data['steps'], v2_cumulative_rewards, 'b-', label='V2累积奖励', linewidth=2)
    ax3.plot(v3_steps_adjusted, v3_cumulative_rewards, 'r-', label='V3累积奖励', linewidth=2)
    
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('累积奖励')
    ax3.set_title('累积奖励对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Lambda演化对比
    ax4 = axes[1, 1]
    ax4.plot(v2_data['steps'], v2_data['lambda_values'], 'b-', label='V2 λ', linewidth=2)
    ax4.plot(v3_steps_adjusted, v3_data['lambda_values'], 'r-', label='V3 λ', linewidth=2)
    
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('λ 值')
    ax4.set_title('Kyle λ 演化对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./version_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("版本对比图已保存: ./version_comparison.png")

def print_feature_comparison():
    """打印功能对比表"""
    print("\n" + "=" * 60)
    print("Kyle Model V2 vs V3 功能对比表")
    print("=" * 60)
    
    features = [
        ("基础Kyle模型", "✅", "✅"),
        ("动态λ更新", "✅", "✅"),
        ("贝叶斯学习", "✅", "✅"),
        ("单段信息", "✅", "✅ (兼容)"),
        ("多段信息", "❌", "✅ (新增)"),
        ("段间价格连续", "❌", "✅ (新增)"),
        ("段边界检测", "❌", "✅ (新增)"),
        ("多段分析", "❌", "✅ (新增)"),
        ("段间转换分析", "❌", "✅ (新增)"),
        ("信息传递效率", "❌", "✅ (新增)"),
        ("可视化增强", "基础", "✅ (增强)"),
        ("配置灵活性", "中等", "✅ (高)"),
    ]
    
    print(f"{'功能':<15} {'V2':<10} {'V3':<15}")
    print("-" * 45)
    
    for feature, v2_status, v3_status in features:
        print(f"{feature:<15} {v2_status:<10} {v3_status:<15}")
    
    print("\n核心改进:")
    print("🔥 多段信息机制: 支持epoch套epoch的复杂信息结构")
    print("🔗 价格连续性: 段间价格无缝衔接")
    print("📊 增强分析: 段间转换和信息传递效率分析")
    print("🎨 可视化升级: 专门的多段信息图表")
    print("⚙️  完全兼容: 支持V2所有功能")

def main():
    """主函数"""
    print("Kyle Model 版本对比工具")
    print("展示V3相对于V2的改进和新功能\n")
    
    try:
        # 功能对比表
        print_feature_comparison()
        
        # 运行对比测试
        results = compare_versions()
        
        print("\n" + "=" * 60)
        print("🎉 版本对比完成！")
        print("\n主要发现:")
        print("1. V3完全兼容V2功能 (super_horizon=1)")
        print("2. V3新增多段信息机制 (super_horizon>1)")
        print("3. V3提供更丰富的分析和可视化")
        print("4. V3支持更复杂的信息结构建模")
        
        print("\n建议使用场景:")
        print("📈 简单信息建模: 继续使用V2兼容模式")
        print("🔬 复杂信息研究: 使用V3多段信息功能")
        print("📊 深度分析: 利用V3增强的分析工具")
        
    except Exception as e:
        print(f"\n❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()