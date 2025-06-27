#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多段真实价值可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import EnhancedInsiderKyleEnv
from visualize import run_single_episode, plot_multi_segment_values, load_model_and_env
import font_config

# 设置中文字体
font_config.setup_chinese_font()

def test_multi_segment_visualization():
    """测试多段信息可视化"""
    print("测试多段信息可视化功能...")
    
    # 测试multi_segment_3配置
    config_name = "multi_segment_3"
    print(f"\n测试配置: {config_name}")
    
    try:
        # 加载模型和环境
        model, env = load_model_and_env(config_name)
        
        # 运行episode收集数据
        data = run_single_episode(model, env)
        
        # 打印收集到的数据信息
        print(f"价格历史长度: {len(data['price_history'])}")
        print(f"段数: {len(set(data.get('segments', [0])))}")
        print(f"段边界: {data.get('segment_boundaries', [])}")
        print(f"段真实价值: {data.get('segment_true_values', [])}")
        print(f"环境value_hist: {getattr(env, 'value_hist', [])}")
        
        # 检查是否为多段信息
        is_multi_segment = len(set(data.get('segments', [0]))) > 1
        print(f"是否为多段信息: {is_multi_segment}")
        
        if is_multi_segment:
            # 绘制多段真实价值图
            plot_multi_segment_values(data, config_name, "./test_plots")
            print(f"多段真实价值图已保存到 ./test_plots/{config_name}_multi_segment_values.png")
        else:
            print("未检测到多段信息，跳过多段可视化")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_manual_multi_segment():
    """手动创建多段环境进行测试"""
    print("\n手动测试多段环境...")
    
    # 创建多段环境
    env = EnhancedInsiderKyleEnv(T=10, super_horizon=3, sigma_u=0.8, sigma_v=1.2)
    
    # 手动运行多个段
    all_prices = []
    all_segments = []
    all_boundaries = []
    all_true_values = []
    
    step_count = 0
    
    for segment in range(3):
        print(f"\n段 {segment + 1}:")
        obs, _ = env.reset()
        print(f"  真实价值: {env.v:.3f}")
        
        # 记录段开始
        if segment == 0:
            all_boundaries.append(0)
        else:
            all_boundaries.append(step_count)
        all_true_values.append(env.v)
        
        # 运行该段的交易
        for t in range(10):  # 每段10轮交易
            action = np.array([np.random.uniform(-1, 1)])  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            all_prices.append(env.current_price)
            all_segments.append(segment)
            step_count += 1
            
            if terminated or truncated:
                break
    
    # 构造数据字典
    manual_data = {
        'price_history': all_prices,
        'segments': all_segments,
        'segment_boundaries': all_boundaries,
        'segment_true_values': all_true_values
    }
    
    print(f"\n手动数据:")
    print(f"价格历史长度: {len(manual_data['price_history'])}")
    print(f"段边界: {manual_data['segment_boundaries']}")
    print(f"段真实价值: {manual_data['segment_true_values']}")
    
    # 绘制图表
    plot_multi_segment_values(manual_data, "manual_test", "./test_plots")
    print(f"手动测试图表已保存到 ./test_plots/manual_test_multi_segment_values.png")

if __name__ == "__main__":
    # 测试现有配置
    test_multi_segment_visualization()
    
    # 手动测试
    test_manual_multi_segment()
    
    print("\n测试完成！")