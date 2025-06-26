#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多段信息分析模块
专门用于分析多段信息配置的特殊性能指标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from font_config import setup_chinese_font

def analyze_segment_transitions(data):
    """
    分析段间转换的特性
    
    Args:
        data: 包含segments和segment_boundaries的数据字典
    
    Returns:
        dict: 段间转换分析结果
    """
    if 'segments' not in data or 'segment_boundaries' not in data:
        return None
    
    segments = np.array(data['segments'])
    boundaries = data['segment_boundaries']
    prices = np.array(data['prices'])
    values = np.array(data['values'])
    
    # 计算段间价格跳跃
    price_jumps = []
    value_changes = []
    
    for boundary in boundaries:
        if boundary > 0 and boundary < len(prices) - 1:
            price_jump = abs(prices[boundary + 1] - prices[boundary])
            value_change = abs(values[boundary + 1] - values[boundary])
            price_jumps.append(price_jump)
            value_changes.append(value_change)
    
    # 计算连续性指标
    avg_price_jump = np.mean(price_jumps) if price_jumps else 0
    max_price_jump = np.max(price_jumps) if price_jumps else 0
    price_continuity_score = 1.0 / (1.0 + avg_price_jump)
    
    return {
        'num_transitions': len(boundaries),
        'avg_price_jump': avg_price_jump,
        'max_price_jump': max_price_jump,
        'price_continuity_score': price_continuity_score,
        'avg_value_change': np.mean(value_changes) if value_changes else 0,
        'price_jumps': price_jumps,
        'value_changes': value_changes
    }

def analyze_segment_efficiency(data):
    """
    分析每个段的信息效率
    
    Args:
        data: 包含段信息的数据字典
    
    Returns:
        dict: 每段效率分析结果
    """
    if 'segments' not in data:
        return None
    
    segments = np.array(data['segments'])
    prices = np.array(data['prices'])
    values = np.array(data['values'])
    conditional_vars = np.array(data['conditional_vars'])
    
    unique_segments = np.unique(segments)
    segment_analysis = {}
    
    for seg in unique_segments:
        seg_mask = segments == seg
        seg_prices = prices[seg_mask]
        seg_values = values[seg_mask]
        seg_vars = conditional_vars[seg_mask]
        
        if len(seg_prices) > 1:
            # 价格收敛性
            price_convergence = 1.0 - (np.std(seg_prices[-5:]) / np.std(seg_prices[:5])) if len(seg_prices) >= 10 else 0
            
            # 信息效率（方差减少）
            initial_var = seg_vars[0] if len(seg_vars) > 0 else 0
            final_var = seg_vars[-1] if len(seg_vars) > 0 else 0
            info_efficiency = (initial_var - final_var) / initial_var if initial_var > 0 else 0
            
            # 价格准确性
            final_price = seg_prices[-1]
            true_value = seg_values[-1]
            price_accuracy = 1.0 / (1.0 + abs(final_price - true_value))
            
            segment_analysis[f'segment_{seg}'] = {
                'length': len(seg_prices),
                'price_convergence': max(0, price_convergence),
                'info_efficiency': max(0, info_efficiency),
                'price_accuracy': price_accuracy,
                'initial_var': initial_var,
                'final_var': final_var,
                'true_value': true_value,
                'final_price': final_price
            }
    
    return segment_analysis

def plot_multi_segment_analysis(data, config_name, save_dir="./plots"):
    """
    绘制多段信息分析图表
    
    Args:
        data: 数据字典
        config_name: 配置名称
        save_dir: 保存目录
    """
    setup_chinese_font()
    os.makedirs(save_dir, exist_ok=True)
    
    if 'segments' not in data or len(set(data['segments'])) <= 1:
        print(f"配置 {config_name} 不是多段信息配置，跳过多段分析")
        return
    
    segments = np.array(data['segments'])
    boundaries = data.get('segment_boundaries', [])
    unique_segments = np.unique(segments)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'多段信息分析 - {config_name}', fontsize=16, fontweight='bold')
    
    steps = range(len(data['prices']))
    
    # 1. 分段价格路径
    ax1 = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_segments)))
    
    for i, seg in enumerate(unique_segments):
        seg_mask = segments == seg
        seg_steps = np.array(steps)[seg_mask]
        seg_prices = np.array(data['prices'])[seg_mask]
        seg_values = np.array(data['values'])[seg_mask]
        
        ax1.plot(seg_steps, seg_prices, color=colors[i], linewidth=2, 
                label=f'段{seg} 价格')
        ax1.axhline(y=seg_values[0], color=colors[i], linestyle='--', alpha=0.7,
                   label=f'段{seg} 真值={seg_values[0]:.2f}')
    
    # 添加段边界
    for boundary in boundaries:
        ax1.axvline(x=boundary, color='red', linestyle=':', alpha=0.8)
    
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('价格')
    ax1.set_title('分段价格演化')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 条件方差分段演化
    ax2 = axes[0, 1]
    for i, seg in enumerate(unique_segments):
        seg_mask = segments == seg
        seg_steps = np.array(steps)[seg_mask]
        seg_vars = np.array(data['conditional_vars'])[seg_mask]
        
        ax2.plot(seg_steps, seg_vars, color=colors[i], linewidth=2,
                label=f'段{seg}')
    
    for boundary in boundaries:
        ax2.axvline(x=boundary, color='red', linestyle=':', alpha=0.8)
    
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('条件方差')
    ax2.set_title('分段条件方差演化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Kyle指标分段演化
    ax3 = axes[1, 0]
    for i, seg in enumerate(unique_segments):
        seg_mask = segments == seg
        seg_steps = np.array(steps)[seg_mask]
        seg_lambdas = np.array(data['lambdas'])[seg_mask]
        
        ax3.plot(seg_steps, seg_lambdas, color=colors[i], linewidth=2,
                label=f'段{seg} λ')
    
    for boundary in boundaries:
        ax3.axvline(x=boundary, color='red', linestyle=':', alpha=0.8)
    
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('λ (价格冲击)')
    ax3.set_title('分段价格冲击演化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 段间转换分析
    ax4 = axes[1, 1]
    transition_analysis = analyze_segment_transitions(data)
    
    if transition_analysis and transition_analysis['price_jumps']:
        price_jumps = transition_analysis['price_jumps']
        value_changes = transition_analysis['value_changes']
        
        ax4.scatter(value_changes, price_jumps, alpha=0.7, s=100)
        ax4.set_xlabel('真值变化幅度')
        ax4.set_ylabel('价格跳跃幅度')
        ax4.set_title('段间转换分析')
        
        # 添加统计信息
        ax4.text(0.05, 0.95, f'平均价格跳跃: {transition_analysis["avg_price_jump"]:.4f}\n'
                            f'连续性得分: {transition_analysis["price_continuity_score"]:.4f}',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, '无段间转换数据', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14)
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{config_name}_multi_segment_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"多段信息分析图已保存: {save_dir}/{config_name}_multi_segment_analysis.png")

def generate_multi_segment_report(data, config_name):
    """
    生成多段信息分析报告
    
    Args:
        data: 数据字典
        config_name: 配置名称
    
    Returns:
        dict: 分析报告
    """
    if 'segments' not in data or len(set(data['segments'])) <= 1:
        return None
    
    print(f"\n{'='*60}")
    print(f"多段信息分析报告: {config_name}")
    print(f"{'='*60}")
    
    segments = np.array(data['segments'])
    unique_segments = np.unique(segments)
    
    print(f"\n基本信息:")
    print(f"总段数: {len(unique_segments)}")
    print(f"总数据点: {len(data['prices'])}")
    print(f"平均每段长度: {len(data['prices']) / len(unique_segments):.1f}")
    
    # 段间转换分析
    transition_analysis = analyze_segment_transitions(data)
    if transition_analysis:
        print(f"\n段间转换分析:")
        print(f"转换次数: {transition_analysis['num_transitions']}")
        print(f"平均价格跳跃: {transition_analysis['avg_price_jump']:.4f}")
        print(f"最大价格跳跃: {transition_analysis['max_price_jump']:.4f}")
        print(f"价格连续性得分: {transition_analysis['price_continuity_score']:.4f}")
        print(f"平均真值变化: {transition_analysis['avg_value_change']:.4f}")
    
    # 各段效率分析
    segment_efficiency = analyze_segment_efficiency(data)
    if segment_efficiency:
        print(f"\n各段效率分析:")
        for seg_name, seg_data in segment_efficiency.items():
            print(f"  {seg_name}:")
            print(f"    长度: {seg_data['length']}")
            print(f"    价格收敛性: {seg_data['price_convergence']:.4f}")
            print(f"    信息效率: {seg_data['info_efficiency']:.4f}")
            print(f"    价格准确性: {seg_data['price_accuracy']:.4f}")
            print(f"    真值: {seg_data['true_value']:.4f}, 最终价格: {seg_data['final_price']:.4f}")
    
    # 整体性能指标
    overall_efficiency = np.mean([seg['info_efficiency'] for seg in segment_efficiency.values()]) if segment_efficiency else 0
    overall_accuracy = np.mean([seg['price_accuracy'] for seg in segment_efficiency.values()]) if segment_efficiency else 0
    
    print(f"\n整体性能:")
    print(f"平均信息效率: {overall_efficiency:.4f}")
    print(f"平均价格准确性: {overall_accuracy:.4f}")
    
    return {
        'transition_analysis': transition_analysis,
        'segment_efficiency': segment_efficiency,
        'overall_efficiency': overall_efficiency,
        'overall_accuracy': overall_accuracy,
        'num_segments': len(unique_segments)
    }

if __name__ == "__main__":
    print("多段信息分析模块")
    print("请通过其他脚本调用此模块的函数")