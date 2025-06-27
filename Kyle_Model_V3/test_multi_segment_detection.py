#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多段信息检测和可视化
"""

import numpy as np
from stable_baselines3 import PPO
from env import EnhancedInsiderKyleEnv
from visualize import run_single_episode, plot_multi_segment_values, load_model_and_env
import font_config

# 设置中文字体
font_config.setup_chinese_font()

def test_multi_segment_detection():
    """测试多段信息检测"""
    print("测试多段信息检测...")
    
    # 测试所有配置
    configs = ["baseline_static", "baseline_dynamic", "high_noise", "low_noise", "long_term", "multi_segment_3", "multi_segment_5"]
    
    for config_name in configs:
        print(f"\n测试配置: {config_name}")
        
        try:
            # 加载模型和环境
            model, env = load_model_and_env(config_name)
            
            # 运行episode收集数据
            data = run_single_episode(model, env)
            
            # 检查数据
            segments = data.get('segments', [])
            unique_segments = set(segments)
            is_multi_segment = len(unique_segments) > 1
            
            print(f"  段数据: {segments[:10]}..." if len(segments) > 10 else f"  段数据: {segments}")
            print(f"  唯一段: {unique_segments}")
            print(f"  是否多段: {is_multi_segment}")
            print(f"  段边界: {data.get('segment_boundaries', [])}")
            print(f"  段真实价值: {data.get('segment_true_values', [])}")
            
            # 如果是多段信息，绘制图表
            if is_multi_segment:
                print(f"  -> 绘制多段真实价值图表")
                plot_multi_segment_values(data, config_name, "./test_plots")
            else:
                print(f"  -> 跳过多段可视化")
                
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

def create_manual_multi_segment_data():
    """手动创建多段数据进行测试"""
    print("\n创建手动多段数据...")
    
    # 模拟3段数据
    price_history = []
    segments = []
    segment_boundaries = [0, 15, 30]
    segment_true_values = [1.2, -0.8, 0.5]
    
    # 生成价格数据
    for seg_idx, (start, true_val) in enumerate(zip(segment_boundaries, segment_true_values)):
        end = segment_boundaries[seg_idx + 1] if seg_idx + 1 < len(segment_boundaries) else 45
        seg_length = end - start
        
        # 生成该段的价格（围绕真实价值波动）
        seg_prices = true_val + np.random.normal(0, 0.1, seg_length)
        price_history.extend(seg_prices)
        segments.extend([seg_idx] * seg_length)
    
    manual_data = {
        'price_history': price_history,
        'segments': segments,
        'segment_boundaries': segment_boundaries,
        'segment_true_values': segment_true_values
    }
    
    print(f"手动数据:")
    print(f"  价格历史长度: {len(manual_data['price_history'])}")
    print(f"  段数据: {set(manual_data['segments'])}")
    print(f"  段边界: {manual_data['segment_boundaries']}")
    print(f"  段真实价值: {manual_data['segment_true_values']}")
    
    # 绘制图表
    plot_multi_segment_values(manual_data, "manual_multi_segment", "./test_plots")
    print(f"手动多段图表已保存")

if __name__ == "__main__":
    # 测试多段信息检测
    test_multi_segment_detection()
    
    # 创建手动多段数据测试
    create_manual_multi_segment_data()
    
    print("\n测试完成！")