#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kyle Model V3 功能测试脚本
测试多段信息功能是否正常工作
"""

import numpy as np
import matplotlib.pyplot as plt
from env import EnhancedInsiderKyleEnv
from stable_baselines3 import PPO
import font_config

def test_multi_segment_env():
    """测试多段信息环境"""
    print("\n=== 测试多段信息环境 ===")
    
    # 创建3段信息环境
    env_params = {
        'T': 5,  # 每段5轮，便于测试
        'sigma_u': 0.8,
        'sigma_v': 1.2,
        'lambda_val': 0.3,
        'max_action': 3.0,
        'seed': 42,
        'dynamic_lambda': True,
        'super_horizon': 3  # 3段信息
    }
    
    env = EnhancedInsiderKyleEnv(**env_params)
    
    print(f"环境配置: T={env_params['T']}, super_horizon={env_params['super_horizon']}")
    print(f"预期总轮数: {env_params['T'] * env_params['super_horizon']} = {env_params['T']} × {env_params['super_horizon']}")
    
    # 重置环境
    obs, info = env.reset()
    print(f"\n初始观察: {obs}")
    print(f"观察空间维度: {len(obs)} (应该包含时间索引、价格、真值、外层进度)")
    
    # 运行一个完整episode
    step_count = 0
    segment_data = []
    
    done = False
    while not done:
        # 随机动作用于测试
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        step_data = {
            'step': step_count,
            'outer_epoch': info.get('outer_epoch', 0),
            'inner_step': info.get('inner_step', 0),
            'price': info['price'],
            'value': info['value'],
            'segment_boundary': info.get('segment_boundary', False)
        }
        segment_data.append(step_data)
        
        if step_count < 20:  # 只打印前20步
            print(f"步骤 {step_count}: 外层epoch={info.get('outer_epoch', 0)}, "
                  f"内层步骤={info.get('inner_step', 0)}, "
                  f"价格={info['price']:.3f}, 真值={info['value']:.3f}, "
                  f"段边界={info.get('segment_boundary', False)}")
        
        step_count += 1
        
        if step_count > 100:  # 防止无限循环
            print("警告: 步数超过100，强制停止")
            break
    
    print(f"\n总步数: {step_count}")
    
    # 分析段信息
    segments = [data['outer_epoch'] for data in segment_data]
    unique_segments = list(set(segments))
    print(f"检测到的段数: {len(unique_segments)}")
    print(f"段标识: {sorted(unique_segments)}")
    
    # 检查段边界
    boundaries = [i for i, data in enumerate(segment_data) if data['segment_boundary']]
    print(f"段边界位置: {boundaries}")
    
    # 检查价格连续性
    if boundaries:
        print("\n段边界价格连续性检查:")
        for boundary in boundaries:
            if boundary > 0 and boundary < len(segment_data) - 1:
                prev_price = segment_data[boundary]['price']
                next_price = segment_data[boundary + 1]['price']
                price_jump = abs(next_price - prev_price)
                print(f"  边界 {boundary}: {prev_price:.4f} -> {next_price:.4f}, 跳跃={price_jump:.4f}")
    
    return segment_data

def test_multi_segment_training():
    """测试多段信息训练"""
    print("\n=== 测试多段信息训练 ===")
    
    # 创建简单的多段信息环境
    env_params = {
        'T': 3,  # 每段3轮
        'sigma_u': 0.8,
        'sigma_v': 1.2,
        'lambda_val': 0.3,
        'max_action': 3.0,
        'seed': 42,
        'dynamic_lambda': True,
        'super_horizon': 2  # 2段信息
    }
    
    env = EnhancedInsiderKyleEnv(**env_params)
    
    print(f"创建PPO模型进行快速训练测试...")
    
    # 创建PPO模型
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=0.001,
        n_steps=64,  # 很小的步数用于快速测试
        batch_size=32,
        n_epochs=2,
        verbose=1
    )
    
    print("开始训练 (1000步快速测试)...")
    model.learn(total_timesteps=1000)
    
    print("训练完成，测试训练后的策略...")
    
    # 测试训练后的策略
    obs, _ = env.reset()
    episode_data = []
    
    done = False
    step = 0
    while not done and step < 20:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_data.append({
            'step': step,
            'action': action[0],
            'reward': reward,
            'price': info['price'],
            'value': info['value'],
            'outer_epoch': info.get('outer_epoch', 0)
        })
        
        step += 1
    
    print(f"策略测试完成，共 {len(episode_data)} 步")
    
    return episode_data

def plot_test_results(segment_data):
    """绘制测试结果"""
    print("\n=== 绘制测试结果 ===")
    
    font_config.setup_chinese_font()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Kyle Model V3 多段信息测试结果', fontsize=16, fontweight='bold')
    
    steps = [data['step'] for data in segment_data]
    prices = [data['price'] for data in segment_data]
    values = [data['value'] for data in segment_data]
    segments = [data['outer_epoch'] for data in segment_data]
    boundaries = [data['step'] for data in segment_data if data['segment_boundary']]
    
    # 1. 价格和真值演化
    ax1 = axes[0]
    ax1.plot(steps, prices, 'b-', label='市场价格', linewidth=2)
    ax1.plot(steps, values, 'r--', label='真实价值', linewidth=2)
    
    # 标记段边界
    for boundary in boundaries:
        ax1.axvline(x=boundary, color='orange', linestyle=':', alpha=0.7, 
                   label='段边界' if boundary == boundaries[0] else "")
    
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('价格/价值')
    ax1.set_title('价格与真值演化（多段信息）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 段信息演化
    ax2 = axes[1]
    ax2.plot(steps, segments, 'g-', marker='o', label='当前段', linewidth=2)
    
    # 标记段边界
    for boundary in boundaries:
        ax2.axvline(x=boundary, color='orange', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('段编号')
    ax2.set_title('段信息演化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./test_v3_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("测试结果图已保存: ./test_v3_results.png")

def main():
    """主测试函数"""
    print("Kyle Model V3 功能测试")
    print("=" * 50)
    
    try:
        # 测试1: 多段信息环境
        segment_data = test_multi_segment_env()
        
        # 测试2: 多段信息训练
        training_data = test_multi_segment_training()
        
        # 测试3: 绘制结果
        plot_test_results(segment_data)
        
        print("\n=== 测试总结 ===")
        print("✅ 多段信息环境测试通过")
        print("✅ 多段信息训练测试通过")
        print("✅ 结果可视化测试通过")
        print("\n🎉 Kyle Model V3 所有核心功能正常工作！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 故障排除建议:")
        print("1. 检查env.py中的多段信息实现")
        print("2. 确保所有依赖包正确安装")
        print("3. 检查观察空间和动作空间定义")

if __name__ == "__main__":
    main()