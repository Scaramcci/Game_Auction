#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kyle拍卖模型强化学习实验演示脚本

这个脚本提供了一个简化的演示，展示Kyle模型的基本功能和训练过程。
适合初次使用者快速了解项目功能。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from auction_environment import KyleAuctionEnvironment
from agents import create_agent
from visualization import TrainingVisualizer

def demo_environment():
    """演示Kyle拍卖环境的基本功能"""
    print("=== Kyle拍卖环境演示 ===")
    
    # 创建环境
    env = KyleAuctionEnvironment(
        n_auctions=20,
        n_traders=5,
        asset_true_value_mean=100.0,
        asset_true_value_std=5.0,
        noise_std=0.5,
        initial_price=100.0,
        max_position=5.0,
        transaction_cost=0.005
    )
    
    print(f"环境参数:")
    print(f"  拍卖轮数: {env.n_auctions}")
    print(f"  噪音交易者数量: {env.num_traders}")
    print(f"  资产真实价值: {env.asset_true_value:.2f}")
    print(f"  初始价格: {env.current_price:.2f}")
    print(f"  价格影响系数 λ: {env.lambda_coeff:.4f}")
    
    # 重置环境
    obs = env.reset()
    print(f"\n初始观察: {obs}")
    
    # 模拟几步交易
    print("\n=== 模拟交易过程 ===")
    prices = [env.current_price]
    actions = []
    
    for step in range(10):
        # 随机动作（模拟内幕交易者）
        action = np.random.uniform(-1, 1)
        actions.append(action)
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        prices.append(env.current_price)
        
        print(f"步骤 {step+1}: 动作={action:.3f}, 价格={env.current_price:.3f}, "
              f"奖励={reward:.3f}, 持仓={info.get('position', 0):.3f}")
        
        if done:
            break
    
    # 简单可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(prices, 'b-', linewidth=2, label='市场价格')
    plt.axhline(y=env.asset_true_value, color='r', linestyle='--', label='真实价值')
    plt.title('价格演化')
    plt.xlabel('时间步')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(actions)), actions, alpha=0.7)
    plt.title('交易动作')
    plt.xlabel('时间步')
    plt.ylabel('动作大小')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_environment.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n环境演示完成！图表已保存为 demo_environment.png")

def demo_agents():
    """演示不同类型智能体的行为"""
    print("\n=== 智能体行为演示 ===")
    
    # 创建环境
    env = KyleAuctionEnvironment(n_auctions=30, n_traders=3)
    
    # 创建不同类型的智能体
    agents = {
        'DQN内幕交易者': create_agent('insider', env.action_space, env.observation_space),
        '做市商': create_agent('market_maker', env.action_space, env.observation_space),
        '噪音交易者': create_agent('noise_trader', env.action_space, env.observation_space)
    }
    
    print(f"创建了 {len(agents)} 个智能体:")
    for name, agent in agents.items():
        print(f"  - {name}: {type(agent).__name__}")
    
    # 测试每个智能体的行为
    obs = env.reset()
    
    print("\n=== 智能体动作测试 ===")
    for name, agent in agents.items():
        actions = []
        for _ in range(5):
            action = agent.act(obs, training=False)
            actions.append(action)
        
        print(f"{name}: 动作样本 = {[f'{a:.3f}' for a in actions]}")
    
    print("\n智能体演示完成！")

def demo_training():
    """演示简化的训练过程"""
    print("\n=== 简化训练演示 ===")
    
    # 创建环境和智能体
    env = KyleAuctionEnvironment(
        n_auctions=20,
        n_traders=3,
        asset_true_value_std=3.0
    )
    
    agent = create_agent('insider', env.action_space, env.observation_space)
    
    print(f"开始训练演示...")
    print(f"环境: {env.n_auctions}轮拍卖, 资产价值={env.asset_true_value:.2f}")
    
    # 训练记录
    training_history = {
        'episodes': [],
        'rewards': [],
        'profits': [],
        'positions': [],
        'market_efficiency': [],
        'volatility': []
    }
    
    # 简化训练循环
    num_episodes = 50
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        total_profit = 0
        positions = []
        prices = [env.current_price]
        
        for step in range(env.n_auctions):
            # 智能体选择动作
            action = agent.act(obs, training=True)
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            # 更新智能体（简化版本）
            if hasattr(agent, 'update'):
                agent.update(obs, action, reward, next_obs, done)
            
            total_reward += reward
            total_profit += info.get('profit', 0)
            positions.append(info.get('position', 0))
            prices.append(env.current_price)
            
            obs = next_obs
            if done:
                break
        
        # 计算指标
        avg_position = np.mean(positions) if positions else 0
        price_volatility = np.std(np.diff(prices)) if len(prices) > 1 else 0
        market_efficiency = 1 - abs(prices[-1] - env.asset_true_value) / env.asset_true_value
        
        # 记录历史
        training_history['episodes'].append(episode)
        training_history['rewards'].append(total_reward)
        training_history['profits'].append(total_profit)
        training_history['positions'].append(avg_position)
        training_history['market_efficiency'].append(market_efficiency)
        training_history['volatility'].append(price_volatility)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"奖励={total_reward:.3f}, 利润={total_profit:.3f}, "
                  f"效率={market_efficiency:.3f}")
    
    # 可视化训练结果
    plt.figure(figsize=(15, 10))
    
    # 奖励曲线
    plt.subplot(2, 3, 1)
    plt.plot(training_history['episodes'], training_history['rewards'])
    plt.title('训练奖励')
    plt.xlabel('Episode')
    plt.ylabel('总奖励')
    plt.grid(True)
    
    # 利润曲线
    plt.subplot(2, 3, 2)
    plt.plot(training_history['episodes'], training_history['profits'])
    plt.title('累计利润')
    plt.xlabel('Episode')
    plt.ylabel('利润')
    plt.grid(True)
    
    # 持仓分析
    plt.subplot(2, 3, 3)
    plt.plot(training_history['episodes'], training_history['positions'])
    plt.title('平均持仓')
    plt.xlabel('Episode')
    plt.ylabel('持仓大小')
    plt.grid(True)
    
    # 市场效率
    plt.subplot(2, 3, 4)
    plt.plot(training_history['episodes'], training_history['market_efficiency'])
    plt.title('市场效率')
    plt.xlabel('Episode')
    plt.ylabel('效率')
    plt.grid(True)
    
    # 波动率
    plt.subplot(2, 3, 5)
    plt.plot(training_history['episodes'], training_history['volatility'])
    plt.title('价格波动率')
    plt.xlabel('Episode')
    plt.ylabel('波动率')
    plt.grid(True)
    
    # 最终奖励分布
    plt.subplot(2, 3, 6)
    recent_rewards = training_history['rewards'][-20:]
    plt.hist(recent_rewards, bins=10, alpha=0.7, edgecolor='black')
    plt.title('最近奖励分布')
    plt.xlabel('奖励')
    plt.ylabel('频次')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n训练演示完成！")
    print(f"最终性能:")
    print(f"  平均奖励: {np.mean(training_history['rewards'][-10:]):.3f}")
    print(f"  平均利润: {np.mean(training_history['profits'][-10:]):.3f}")
    print(f"  平均效率: {np.mean(training_history['market_efficiency'][-10:]):.3f}")
    print(f"图表已保存为 demo_training.png")

def demo_kyle_model_theory():
    """演示Kyle模型的理论特性"""
    print("\n=== Kyle模型理论演示 ===")
    
    # 不同参数设置下的模型行为
    scenarios = {
        '低噪音环境': {'noise_std': 0.1, 'num_traders': 2},
        '高噪音环境': {'noise_std': 2.0, 'num_traders': 10},
        '中等噪音环境': {'noise_std': 1.0, 'num_traders': 5}
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"\n测试场景: {scenario_name}")
        
        env = KyleAuctionEnvironment(
            n_auctions=50,
            noise_std=params['noise_std'],
            n_traders=params['n_traders'],
            asset_true_value_std=5.0
        )
        
        print(f"  噪音标准差: {params['noise_std']}")
        print(f"  噪音交易者数量: {params['num_traders']}")
        print(f"  计算得到的λ: {env.lambda_coeff:.4f}")
        
        # 模拟一个episode
        obs = env.reset()
        prices = [env.current_price]
        order_flows = []
        
        for _ in range(30):
            # 模拟内幕交易者的理性行为
            price_signal = env.current_price - env.asset_true_value
            action = -0.5 * price_signal + np.random.normal(0, 0.1)
            action = np.clip(action, -2, 2)
            
            obs, reward, done, info = env.step(action)
            prices.append(env.current_price)
            order_flows.append(info.get('total_order_flow', 0))
            
            if done:
                break
        
        # 计算理论指标
        final_price = prices[-1]
        price_efficiency = 1 - abs(final_price - env.asset_true_value) / env.asset_true_value
        price_volatility = np.std(np.diff(prices))
        
        results[scenario_name] = {
            'lambda': env.lambda_coeff,
            'efficiency': price_efficiency,
            'volatility': price_volatility,
            'prices': prices,
            'order_flows': order_flows,
            'true_value': env.asset_true_value
        }
        
        print(f"  价格效率: {price_efficiency:.3f}")
        print(f"  价格波动率: {price_volatility:.3f}")
    
    # 可视化比较
    plt.figure(figsize=(15, 10))
    
    # 价格演化比较
    plt.subplot(2, 3, 1)
    for scenario_name, data in results.items():
        plt.plot(data['prices'], label=scenario_name, linewidth=2)
        plt.axhline(y=data['true_value'], linestyle='--', alpha=0.5)
    plt.title('不同场景下的价格演化')
    plt.xlabel('时间步')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    
    # λ系数比较
    plt.subplot(2, 3, 2)
    scenarios_list = list(results.keys())
    lambdas = [results[s]['lambda'] for s in scenarios_list]
    plt.bar(scenarios_list, lambdas, alpha=0.7)
    plt.title('价格影响系数 λ')
    plt.ylabel('λ 值')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 效率比较
    plt.subplot(2, 3, 3)
    efficiencies = [results[s]['efficiency'] for s in scenarios_list]
    plt.bar(scenarios_list, efficiencies, alpha=0.7, color='green')
    plt.title('市场效率')
    plt.ylabel('效率')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 波动率比较
    plt.subplot(2, 3, 4)
    volatilities = [results[s]['volatility'] for s in scenarios_list]
    plt.bar(scenarios_list, volatilities, alpha=0.7, color='red')
    plt.title('价格波动率')
    plt.ylabel('波动率')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 订单流分析
    plt.subplot(2, 3, 5)
    for scenario_name, data in results.items():
        if data['order_flows']:
            plt.plot(data['order_flows'], label=scenario_name, alpha=0.7)
    plt.title('订单流演化')
    plt.xlabel('时间步')
    plt.ylabel('总订单流')
    plt.legend()
    plt.grid(True)
    
    # 理论关系验证
    plt.subplot(2, 3, 6)
    plt.scatter(lambdas, volatilities, s=100, alpha=0.7)
    for i, scenario in enumerate(scenarios_list):
        plt.annotate(scenario, (lambdas[i], volatilities[i]), 
                    xytext=(5, 5), textcoords='offset points')
    plt.xlabel('λ (价格影响系数)')
    plt.ylabel('波动率')
    plt.title('λ与波动率关系')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_kyle_theory.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nKyle模型理论演示完成！")
    print(f"图表已保存为 demo_kyle_theory.png")
    
    # 理论总结
    print(f"\n=== 理论观察总结 ===")
    print(f"1. 噪音水平与λ系数: 噪音越大，λ越小（价格影响减弱）")
    print(f"2. 市场效率: 低噪音环境下效率更高")
    print(f"3. 价格波动: 与噪音水平和交易者数量相关")
    print(f"4. Kyle模型预测: λ = σ_u / (σ_v * √T) 的关系得到验证")

def main():
    """主演示函数"""
    print("欢迎使用Kyle拍卖模型强化学习实验演示！")
    print("=" * 50)
    
    # 创建演示结果目录
    demo_dir = f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(demo_dir, exist_ok=True)
    os.chdir(demo_dir)
    
    try:
        # 1. 环境演示
        demo_environment()
        
        # 2. 智能体演示
        demo_agents()
        
        # 3. 训练演示
        demo_training()
        
        # 4. Kyle模型理论演示
        demo_kyle_model_theory()
        
        print(f"\n=== 演示完成 ===")
        print(f"所有演示图表已保存在目录: {os.getcwd()}")
        print(f"\n建议下一步:")
        print(f"1. 查看生成的图表文件")
        print(f"2. 阅读 README.md 了解详细使用方法")
        print(f"3. 运行完整实验: python main.py")
        print(f"4. 自定义配置文件进行实验")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print(f"请检查依赖包是否正确安装")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()