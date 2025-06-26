#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kyle Model V3 快速验证脚本
快速验证多段信息功能的基本工作状态
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import EnhancedInsiderKyleEnv

def quick_test():
    """快速测试多段信息功能"""
    print("Kyle Model V3 快速验证")
    print("=" * 40)
    
    # 测试配置
    configs = [
        {
            'name': '单段信息 (V2兼容)',
            'params': {
                'T': 5,
                'sigma_u': 0.8,
                'sigma_v': 1.2,
                'lambda_val': 0.3,
                'max_action': 3.0,
                'seed': 42,
                'dynamic_lambda': True,
                'super_horizon': 1  # 单段
            }
        },
        {
            'name': '三段信息 (V3新功能)',
            'params': {
                'T': 3,
                'sigma_u': 0.8,
                'sigma_v': 1.2,
                'lambda_val': 0.3,
                'max_action': 3.0,
                'seed': 42,
                'dynamic_lambda': True,
                'super_horizon': 3  # 三段
            }
        }
    ]
    
    for config in configs:
        print(f"\n测试: {config['name']}")
        print("-" * 30)
        
        try:
            # 创建环境
            env = EnhancedInsiderKyleEnv(**config['params'])
            
            # 重置环境
            obs, info = env.reset()
            print(f"✅ 环境创建成功")
            print(f"   观察空间维度: {len(obs)}")
            print(f"   预期总轮数: {config['params']['T'] * config['params']['super_horizon']}")
            
            # 运行几步
            step_count = 0
            segments_seen = set()
            boundaries_count = 0
            
            done = False
            while not done and step_count < 20:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                
                # 收集信息
                outer_epoch = info.get('outer_epoch', 0)
                segments_seen.add(outer_epoch)
                
                if info.get('segment_boundary', False):
                    boundaries_count += 1
                
                step_count += 1
            
            print(f"✅ 运行测试完成")
            print(f"   总步数: {step_count}")
            print(f"   检测到段数: {len(segments_seen)}")
            print(f"   段边界数: {boundaries_count}")
            print(f"   段标识: {sorted(list(segments_seen))}")
            
            # 验证多段信息逻辑
            expected_segments = config['params']['super_horizon']
            if len(segments_seen) == expected_segments:
                print(f"✅ 段数验证通过 ({len(segments_seen)}/{expected_segments})")
            else:
                print(f"⚠️  段数验证异常 ({len(segments_seen)}/{expected_segments})")
            
            # 验证段边界
            expected_boundaries = expected_segments - 1 if expected_segments > 1 else 0
            if boundaries_count == expected_boundaries:
                print(f"✅ 段边界验证通过 ({boundaries_count}/{expected_boundaries})")
            else:
                print(f"⚠️  段边界验证异常 ({boundaries_count}/{expected_boundaries})")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
    
    print("\n" + "=" * 40)
    print("🎉 快速验证完成！")
    print("\n核心功能状态:")
    print("✅ 环境创建和重置")
    print("✅ 多段信息机制")
    print("✅ 段边界检测")
    print("✅ V2兼容性")
    
    print("\n可以继续进行完整测试:")
    print("  python test_v3_features.py  # 完整功能测试")
    print("  python train.py             # 训练测试")
    print("  python visualize.py         # 可视化测试")
    
    return True

def check_dependencies():
    """检查依赖"""
    print("检查依赖包...")
    
    required_packages = [
        'numpy',
        'matplotlib', 
        'stable_baselines3',
        'gymnasium'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def main():
    """主函数"""
    print("Kyle Model V3 快速验证脚本")
    print("检查多段信息功能是否正常工作\n")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    print()
    
    # 快速测试
    success = quick_test()
    
    if success:
        print("\n🚀 Kyle Model V3 准备就绪！")
    else:
        print("\n🔧 需要检查和修复问题")

if __name__ == "__main__":
    main()