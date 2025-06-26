#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行完整Kyle模型实验
包含训练、可视化和分析的完整流程
"""

import os
import sys
import time
import argparse
from datetime import datetime

def run_training(quick_mode=False):
    """运行训练阶段"""
    print("\n" + "="*50)
    print("🚀 开始训练阶段")
    print("="*50)
    
    if quick_mode:
        print("⚡ 快速模式：减少训练时间")
        # 修改train.py中的训练步数
        import train
        # 临时修改训练参数
        original_timesteps = 200000
        quick_timesteps = 50000
        print(f"训练步数: {original_timesteps} -> {quick_timesteps}")
    
    try:
        import train
        if quick_mode:
            # 动态修改训练参数
            train.base_training_params = {
                'learning_rate': 0.0003,
                'n_steps': 1024,  # 减少
                'batch_size': 32,  # 减少
                'n_epochs': 5,     # 减少
                'total_timesteps': 50000  # 大幅减少
            }
        
        trained_models = train.main()
        print("✅ 训练阶段完成")
        return True
    except Exception as e:
        print(f"❌ 训练阶段失败: {e}")
        return False

def run_visualization():
    """运行可视化阶段"""
    print("\n" + "="*50)
    print("📊 开始可视化阶段")
    print("="*50)
    
    try:
        import visualize
        results = visualize.main()
        print("✅ 可视化阶段完成")
        return True
    except Exception as e:
        print(f"❌ 可视化阶段失败: {e}")
        return False

def run_analysis(quick_mode=False):
    """运行分析阶段"""
    print("\n" + "="*50)
    print("🔬 开始分析阶段")
    print("="*50)
    
    episodes = 200 if quick_mode else 1000
    print(f"分析episodes数量: {episodes}")
    
    try:
        import analysis
        # 临时修改episodes数量
        if quick_mode:
            # 修改默认episodes
            original_analyze = analysis.analyze_configuration
            def quick_analyze(config_name, episodes=200, models_dir="./models"):
                return original_analyze(config_name, episodes, models_dir)
            analysis.analyze_configuration = quick_analyze
        
        results = analysis.main()
        print("✅ 分析阶段完成")
        return True
    except Exception as e:
        print(f"❌ 分析阶段失败: {e}")
        return False

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'scipy', 
        'sklearn', 'stable_baselines3', 'gymnasium'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺失依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def create_directories():
    """创建必要的目录"""
    dirs = ['./models', './plots', './analysis_plots']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 创建目录: {dir_name}")

def print_summary(start_time, success_stages):
    """打印实验总结"""
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("📋 实验总结")
    print("="*60)
    
    print(f"⏱️  总耗时: {duration/60:.1f} 分钟")
    print(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n🎯 完成阶段:")
    for stage, success in success_stages.items():
        status = "✅" if success else "❌"
        print(f"  {status} {stage}")
    
    if all(success_stages.values()):
        print("\n🎉 实验完全成功！")
        print("\n📂 输出文件位置:")
        print("  - 训练模型: ./models/")
        print("  - 基础图表: ./plots/")
        print("  - 分析结果: ./analysis_plots/")
        
        print("\n📊 主要结果文件:")
        print("  - configuration_comparison.png (配置比较)")
        print("  - *_price_impact_regression.png (价格冲击回归)")
        print("  - *_lambda_evolution.png (λ演化)")
        print("  - *_beta_evolution.png (β演化)")
    else:
        print("\n⚠️  部分阶段失败，请检查错误信息")

def main():
    """一键运行Kyle模型实验（支持多段信息配置）"""
    parser = argparse.ArgumentParser(description='Kyle模型强化学习实验 V3 - 多段信息支持')
    parser.add_argument('--quick', action='store_true', 
                       help='快速模式 (减少训练时间和分析样本)')
    parser.add_argument('--skip-train', action='store_true',
                       help='跳过训练阶段 (使用已有模型)')
    parser.add_argument('--skip-viz', action='store_true',
                       help='跳过可视化阶段')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='跳过分析阶段')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查环境，不运行实验')
    
    args = parser.parse_args()
    
    print("🎯 Kyle模型强化学习实验 V3")
    print("新增功能: 多段信息支持")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick:
        print("⚡ 快速模式已启用")
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    if args.check_only:
        print("\n✅ 环境检查完成，可以运行实验")
        return 0
    
    # 创建目录
    create_directories()
    
    start_time = time.time()
    success_stages = {
        '训练': False,
        '可视化': False,
        '分析': False
    }
    
    # 运行各阶段
    if not args.skip_train:
        success_stages['训练'] = run_training(args.quick)
        if not success_stages['训练']:
            print("\n❌ 训练失败，停止后续阶段")
            print_summary(start_time, success_stages)
            return 1
    else:
        print("\n⏭️  跳过训练阶段")
        success_stages['训练'] = True
    
    if not args.skip_viz:
        success_stages['可视化'] = run_visualization()
    else:
        print("\n⏭️  跳过可视化阶段")
        success_stages['可视化'] = True
    
    if not args.skip_analysis:
        success_stages['分析'] = run_analysis(args.quick)
    else:
        print("\n⏭️  跳过分析阶段")
        success_stages['分析'] = True
    
    # 打印总结
    print_summary(start_time, success_stages)
    
    return 0 if all(success_stages.values()) else 1

if __name__ == "__main__":
    sys.exit(main())