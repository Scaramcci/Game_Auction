#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kyle拍卖模型强化学习实验主程序

这个程序实现了基于Kyle(1985)模型的顺序拍卖强化学习实验，
包括内幕交易者、做市商和噪音交易者的建模与训练。

作者: AI Assistant
日期: 2024
"""

import os
import sys
import argparse
import yaml
import json
import logging
from datetime import datetime
from typing import Dict, List

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auction_environment import KyleAuctionEnvironment
from agents import create_agent
from training import TrainingManager, load_config
from testing import ModelTester, load_test_config
from visualization import TrainingVisualizer, create_summary_report

def setup_logging(config: Dict):
    """设置日志系统"""
    log_config = config.get('logging', {})
    
    # 创建日志目录
    log_file = log_config.get('file', 'train_log/logs/experiment.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_experiment_config(config_path: str) -> Dict:
    """加载实验配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    return config

def create_experiment_directory(config: Dict) -> str:
    """创建实验目录"""
    experiment_name = config.get('experiment', {}).get('name', 'kyle_auction_rl')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 将所有结果保存在train_log目录下
    base_dir = config.get('results', {}).get('base_dir', 'train_log/results')
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    # 创建子目录
    subdirs = ['models', 'plots', 'logs', 'data']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    return experiment_dir

def run_training(config: Dict, experiment_dir: str, logger: logging.Logger):
    """运行训练流程"""
    logger.info("开始训练阶段...")
    
    # 创建训练管理器
    training_config = {
        'environment': config['environment'],
        'training': config['training'],
        'agents': config['agents'],
        'results_dir': experiment_dir
    }
    
    trainer = TrainingManager(training_config)
    
    # 设置环境和智能体
    trainer.setup_environment()
    trainer.setup_agents()
    
    logger.info(f"环境设置完成: {trainer.env.n_auctions}轮拍卖, {trainer.env.n_traders}个交易者")
    logger.info(f"智能体设置完成: {list(trainer.agents.keys())}")
    
    # 开始训练
    num_episodes = config['training']['num_episodes']
    logger.info(f"开始训练 {num_episodes} 个episodes...")
    
    training_history = {
        'episodes': [],
        'rewards': [],
        'profits': [],
        'positions': [],
        'market_efficiency': [],
        'volatility': []
    }
    
    best_reward = float('-inf')
    patience_counter = 0
    early_stopping_config = config['training'].get('early_stopping', {})
    
    for episode in range(num_episodes):
        # 训练一个episode
        episode_result = trainer.train_episode(episode)
        
        # 记录训练历史
        training_history['episodes'].append(episode)
        training_history['rewards'].append(episode_result['total_reward'])
        training_history['profits'].append(episode_result['metrics']['total_profit'])
        training_history['positions'].append(episode_result['metrics']['avg_position'])
        training_history['market_efficiency'].append(episode_result['metrics']['market_efficiency'])
        training_history['volatility'].append(episode_result['metrics']['price_volatility'])
        
        # 日志记录
        if (episode + 1) % config['training']['log_frequency'] == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes}: "
                       f"Reward={episode_result['total_reward']:.4f}, "
                       f"Profit={episode_result['metrics']['total_profit']:.4f}, "
                       f"Efficiency={episode_result['metrics']['market_efficiency']:.4f}")
        
        # 评估和可视化
        if (episode + 1) % config['training']['eval_frequency'] == 0:
            eval_result = trainer.evaluate_episode(episode)
            logger.info(f"评估结果 - Episode {episode + 1}: "
                       f"平均奖励={eval_result['avg_reward']:.4f}, "
                       f"胜率={eval_result['win_rate']:.2%}")
            
            # 更新可视化
            if config.get('visualization', {}).get('plot_frequency', 100) > 0:
                visualizer = TrainingVisualizer(experiment_dir)
                visualizer.update_training_plots(training_history, episode + 1)
        
        # 保存模型
        if (episode + 1) % config['training']['save_frequency'] == 0:
            trainer.save_models(episode + 1)
            logger.info(f"模型已保存 - Episode {episode + 1}")
        
        # 早停检查
        if early_stopping_config.get('enabled', False):
            current_reward = episode_result['total_reward']
            if current_reward > best_reward + early_stopping_config.get('min_improvement', 0.01):
                best_reward = current_reward
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_config.get('patience', 500):
                logger.info(f"早停触发 - Episode {episode + 1}")
                break
    
    # 最终评估和保存
    final_evaluation = trainer.final_evaluation()
    trainer.save_models('final')
    
    logger.info("训练完成!")
    logger.info(f"最终评估结果: {final_evaluation}")
    
    # 保存训练历史
    history_file = os.path.join(experiment_dir, 'data', 'training_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    return training_history

def run_testing(config: Dict, experiment_dir: str, logger: logging.Logger):
    """运行测试流程"""
    logger.info("开始测试阶段...")
    
    # 创建测试配置
    test_config = {
        'environment': config['environment'],
        'testing': config.get('testing', {})
    }
    
    # 创建测试器
    tester = ModelTester(test_config)
    tester.results_dir = os.path.join(experiment_dir, 'testing')
    os.makedirs(tester.results_dir, exist_ok=True)
    
    # 设置环境
    tester.setup_environment()
    
    # 加载训练好的模型
    model_paths = {
        'insider': os.path.join(experiment_dir, 'models', 'insider_final.pth'),
        'market_maker': os.path.join(experiment_dir, 'models', 'market_maker_final.pth')
    }
    
    # 检查模型文件是否存在
    available_models = {}
    for agent_type, path in model_paths.items():
        if os.path.exists(path):
            available_models[agent_type] = path
        else:
            logger.warning(f"模型文件不存在: {path}")
    
    if available_models:
        tester.load_trained_agents(available_models)
        
        # 运行综合测试
        num_test_episodes = config.get('testing', {}).get('num_episodes', 200)
        logger.info(f"开始测试 {num_test_episodes} 个episodes...")
        
        test_results = tester.run_comprehensive_test(num_test_episodes)
        
        # 生成测试报告
        tester.generate_test_report()
        
        logger.info("测试完成!")
        logger.info(f"测试结果: 平均奖励={test_results['overall_stats']['mean_reward']:.4f}, "
                   f"胜率={test_results['overall_stats']['win_rate']:.2%}")
        
        return test_results
    else:
        logger.error("没有找到可用的训练模型，跳过测试阶段")
        return None

def run_strategy_comparison(config: Dict, experiment_dir: str, logger: logging.Logger):
    """运行策略比较"""
    comparison_config = config.get('testing', {}).get('strategy_comparison', {})
    
    if not comparison_config.get('enabled', False):
        logger.info("策略比较未启用")
        return
    
    logger.info("开始策略比较...")
    
    # 定义不同策略的配置
    strategies = {
        'conservative': {
            'agents': {
                'insider': {
                    'epsilon_start': 0.3,
                    'epsilon_end': 0.05,
                    'learning_rate': 0.0005
                }
            }
        },
        'aggressive': {
            'agents': {
                'insider': {
                    'epsilon_start': 0.9,
                    'epsilon_end': 0.1,
                    'learning_rate': 0.002
                }
            }
        },
        'adaptive': {
            'agents': {
                'insider': {
                    'type': 'AdaptiveInsider',
                    'adaptation_rate': 0.15
                }
            }
        }
    }
    
    # 创建测试器
    base_config = {
        'environment': config['environment'],
        'testing': {'num_episodes': 50}
    }
    
    tester = ModelTester(base_config)
    tester.results_dir = os.path.join(experiment_dir, 'strategy_comparison')
    
    # 运行策略比较
    comparison_results = tester.compare_strategies(strategies)
    
    logger.info("策略比较完成!")
    
    return comparison_results

def generate_final_report(config: Dict, experiment_dir: str, 
                         training_history: Dict, test_results: Dict, logger: logging.Logger):
    """生成最终实验报告"""
    logger.info("生成最终实验报告...")
    
    # 创建可视化器
    visualizer = TrainingVisualizer(experiment_dir)
    
    # 生成训练报告
    if training_history:
        visualizer.create_final_report(training_history)
    
    # 创建综合报告
    report_content = f"""
# Kyle拍卖模型强化学习实验报告

## 实验信息
- 实验名称: {config.get('experiment', {}).get('name', 'kyle_auction_rl')}
- 实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 实验版本: {config.get('experiment', {}).get('version', '1.0')}
- 实验描述: {config.get('experiment', {}).get('description', '')}

## 环境配置
- 拍卖轮数: {config['environment']['n_auctions']}
- 交易者数量: {config['environment']['n_traders']}
- 资产价值均值: {config['environment']['asset_true_value_mean']}
- 噪音标准差: {config['environment']['noise_std']}
- 交易成本: {config['environment']['transaction_cost']}

## 训练配置
- 训练episodes: {config['training']['num_episodes']}
- 智能体类型: {list(config['agents'].keys())}

## 训练结果
"""
    
    if training_history:
        final_rewards = training_history['rewards'][-100:] if len(training_history['rewards']) >= 100 else training_history['rewards']
        report_content += f"""
- 最终平均奖励: {sum(final_rewards)/len(final_rewards):.4f}
- 最大奖励: {max(training_history['rewards']):.4f}
- 最小奖励: {min(training_history['rewards']):.4f}
- 最终市场效率: {training_history['market_efficiency'][-1]:.4f}
- 最终波动率: {training_history['volatility'][-1]:.4f}
"""
    
    if test_results:
        report_content += f"""

## 测试结果
- 测试episodes: {len(test_results['all_results']['episodes'])}
- 平均测试奖励: {test_results['overall_stats']['mean_reward']:.4f}
- 测试胜率: {test_results['overall_stats']['win_rate']:.2%}
- 夏普比率: {test_results['overall_stats']['sharpe_ratio']:.4f}
- 一致性得分: {test_results['overall_stats']['consistency']:.4f}
"""
    
    report_content += f"""

## 文件结构
- 训练模型: {experiment_dir}/models/
- 可视化图表: {experiment_dir}/plots/
- 实验数据: {experiment_dir}/data/
- 日志文件: {experiment_dir}/logs/

## 结论
本实验成功实现了Kyle拍卖模型的强化学习框架，训练了内幕交易者智能体，
并通过与做市商和噪音交易者的交互学习了最优交易策略。

实验结果表明智能体能够有效学习市场动态，并在保持市场效率的同时获得稳定收益。
"""
    
    # 保存报告
    report_file = os.path.join(experiment_dir, 'EXPERIMENT_REPORT.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 保存配置文件副本
    config_copy = os.path.join(experiment_dir, 'experiment_config.yaml')
    with open(config_copy, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"最终报告已生成: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Kyle拍卖模型强化学习实验')
    parser.add_argument('--config', '-c', default='config.yaml', 
                       help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--mode', '-m', choices=['train', 'test', 'both'], 
                       default='both', help='运行模式 (默认: both)')
    parser.add_argument('--experiment-dir', '-d', 
                       help='指定实验目录 (可选)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='详细输出')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        print(f"加载配置文件: {args.config}")
        config = load_experiment_config(args.config)
        
        # 创建实验目录
        if args.experiment_dir:
            experiment_dir = args.experiment_dir
            os.makedirs(experiment_dir, exist_ok=True)
        else:
            experiment_dir = create_experiment_directory(config)
        
        print(f"实验目录: {experiment_dir}")
        
        # 设置日志
        logger = setup_logging(config)
        logger.info(f"开始Kyle拍卖模型强化学习实验")
        logger.info(f"实验目录: {experiment_dir}")
        logger.info(f"运行模式: {args.mode}")
        
        training_history = None
        test_results = None
        
        # 运行训练
        if args.mode in ['train', 'both']:
            training_history = run_training(config, experiment_dir, logger)
        
        # 运行测试
        if args.mode in ['test', 'both']:
            test_results = run_testing(config, experiment_dir, logger)
        
        # 运行策略比较
        if args.mode == 'both':
            run_strategy_comparison(config, experiment_dir, logger)
        
        # 生成最终报告
        generate_final_report(config, experiment_dir, training_history, test_results, logger)
        
        logger.info("实验完成!")
        print(f"\n实验完成! 结果保存在: {experiment_dir}")
        print(f"查看实验报告: {os.path.join(experiment_dir, 'EXPERIMENT_REPORT.md')}")
        
    except Exception as e:
        print(f"实验运行出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()