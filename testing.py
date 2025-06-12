import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime

from auction_environment import KyleAuctionEnvironment
from agents import create_agent, BaseAgent
from visualization import TrainingVisualizer

class ModelTester:
    """模型测试器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.env = None
        self.agents = {}
        self.test_results = {}
        
        # 创建结果目录
        self.results_dir = os.path.join('results', 'testing', 
                                       datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化可视化器
        self.visualizer = TrainingVisualizer(self.results_dir)
    
    def setup_environment(self):
        """设置测试环境"""
        env_config = self.config.get('environment', {})
        self.env = KyleAuctionEnvironment(
            n_auctions=env_config.get('n_auctions', 100),
            n_traders=env_config.get('n_traders', 10),
            asset_true_value_mean=env_config.get('asset_true_value_mean', 100.0),
            asset_true_value_std=env_config.get('asset_true_value_std', 10.0),
            noise_std=env_config.get('noise_std', 1.0),
            initial_price=env_config.get('initial_price', 100.0),
            max_position=env_config.get('max_position', 10.0),
            transaction_cost=env_config.get('transaction_cost', 0.01)
        )
        print(f"Test environment setup complete")
    
    def load_trained_agents(self, model_paths: Dict[str, str]):
        """加载训练好的智能体"""
        for agent_type, model_path in model_paths.items():
            if os.path.exists(model_path):
                agent = create_agent(agent_type, self.env.action_space, 
                                   self.env.observation_space)
                
                # 加载模型权重
                if hasattr(agent, 'load_model'):
                    agent.load_model(model_path)
                    print(f"Loaded {agent_type} model from {model_path}")
                
                self.agents[agent_type] = agent
            else:
                print(f"Warning: Model file {model_path} not found for {agent_type}")
    
    def run_test_episode(self, episode_num: int = 0, record_details: bool = True) -> Dict:
        """运行单个测试episode"""
        obs = self.env.reset()
        done = False
        step = 0
        
        # 记录详细数据
        episode_data = {
            'prices': [self.env.current_price],
            'actions': [],
            'positions': [],
            'rewards': [],
            'order_flows': [],
            'market_states': [],
            'agent_states': {agent_type: [] for agent_type in self.agents.keys()}
        }
        
        total_rewards = {agent_type: 0 for agent_type in self.agents.keys()}
        
        while not done:
            actions = {}
            
            # 获取每个智能体的动作
            for agent_type, agent in self.agents.items():
                agent_obs = self.env.get_observation(0)  # 假设测试主要智能体
                action = agent.act(agent_obs, training=False)
                actions[agent_type] = action
                
                if record_details:
                    episode_data['agent_states'][agent_type].append({
                        'observation': agent_obs.copy(),
                        'action': action
                    })
            
            # 执行主要智能体的动作（通常是insider）
            main_action = actions.get('insider', 0)
            # 将动作包装成列表传递给环境
            obs, reward, done, info = self.env.step([main_action])
            
            total_rewards['insider'] += reward
            
            if record_details:
                episode_data['actions'].append(main_action)
                episode_data['positions'].append(info.get('position', 0))
                episode_data['rewards'].append(reward)
                episode_data['prices'].append(self.env.current_price)
                episode_data['order_flows'].append(info.get('total_order_flow', 0))
                episode_data['market_states'].append(self.env.get_market_state())
            
            step += 1
        
        # 计算episode统计
        episode_stats = self._calculate_episode_stats(episode_data, total_rewards)
        
        return {
            'episode_data': episode_data,
            'episode_stats': episode_stats,
            'total_rewards': total_rewards
        }
    
    def run_comprehensive_test(self, num_episodes: int = 100) -> Dict:
        """运行综合测试"""
        print(f"Starting comprehensive test with {num_episodes} episodes...")
        
        all_results = {
            'episodes': [],
            'rewards': [],
            'profits': [],
            'positions': [],
            'market_efficiency': [],
            'volatility': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': []
        }
        
        detailed_episodes = []  # 保存详细的episode数据
        
        for episode in range(num_episodes):
            # 每10个episode记录详细数据
            record_details = (episode % 10 == 0)
            
            result = self.run_test_episode(episode, record_details)
            
            # 收集统计数据
            all_results['episodes'].append(episode)
            all_results['rewards'].append(result['total_rewards']['insider'])
            all_results['profits'].append(result['episode_stats']['total_profit'])
            all_results['positions'].append(result['episode_stats']['avg_position'])
            all_results['market_efficiency'].append(result['episode_stats']['market_efficiency'])
            all_results['volatility'].append(result['episode_stats']['price_volatility'])
            all_results['sharpe_ratios'].append(result['episode_stats']['sharpe_ratio'])
            all_results['max_drawdowns'].append(result['episode_stats']['max_drawdown'])
            all_results['win_rates'].append(result['episode_stats']['win_rate'])
            
            if record_details:
                detailed_episodes.append(result)
            
            if (episode + 1) % 20 == 0:
                print(f"Completed {episode + 1}/{num_episodes} test episodes")
        
        # 计算总体统计
        overall_stats = self._calculate_overall_stats(all_results)
        
        # 保存结果
        self.test_results = {
            'all_results': all_results,
            'detailed_episodes': detailed_episodes,
            'overall_stats': overall_stats,
            'config': self.config
        }
        
        return self.test_results
    
    def _calculate_episode_stats(self, episode_data: Dict, total_rewards: Dict) -> Dict:
        """计算单个episode的统计数据"""
        prices = episode_data['prices']
        actions = episode_data['actions']
        positions = episode_data['positions']
        rewards = episode_data['rewards']
        
        # 基本统计
        total_profit = sum(rewards)
        avg_position = np.mean(positions) if positions else 0
        
        # 价格波动率
        if len(prices) > 1:
            price_returns = np.diff(prices) / prices[:-1]
            price_volatility = np.std(price_returns)
        else:
            price_volatility = 0
        
        # 市场效率（价格与真实价值的偏差）
        market_state = episode_data['market_states'][0] if episode_data['market_states'] else {}
        true_value = market_state.get('asset_true_value', prices[0])
        
        if len(prices) > 1:
            final_price = prices[-1]
            market_efficiency = 1 - abs(final_price - true_value) / true_value
        else:
            market_efficiency = 0
        
        # 夏普比率
        if len(rewards) > 1 and np.std(rewards) > 0:
            sharpe_ratio = np.mean(rewards) / np.std(rewards)
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        if rewards:
            cumulative_rewards = np.cumsum(rewards)
            running_max = np.maximum.accumulate(cumulative_rewards)
            drawdowns = (cumulative_rewards - running_max) / (running_max + 1e-8)
            max_drawdown = np.min(drawdowns)
        else:
            max_drawdown = 0
        
        # 胜率
        win_rate = np.mean([r > 0 for r in rewards]) if rewards else 0
        
        return {
            'total_profit': total_profit,
            'avg_position': avg_position,
            'price_volatility': price_volatility,
            'market_efficiency': market_efficiency,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len([a for a in actions if abs(a) > 0.01]),
            'avg_trade_size': np.mean([abs(a) for a in actions if abs(a) > 0.01]) if actions else 0
        }
    
    def _calculate_overall_stats(self, all_results: Dict) -> Dict:
        """计算总体统计数据"""
        rewards = all_results['rewards']
        profits = all_results['profits']
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_profit': np.mean(profits),
            'std_profit': np.std(profits),
            'total_profit': sum(profits),
            'win_rate': np.mean([r > 0 for r in rewards]),
            'sharpe_ratio': np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0,
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'mean_market_efficiency': np.mean(all_results['market_efficiency']),
            'mean_volatility': np.mean(all_results['volatility']),
            'consistency': 1 - np.std(rewards) / (abs(np.mean(rewards)) + 1e-8)
        }
    
    def generate_test_report(self):
        """生成测试报告"""
        if not self.test_results:
            print("No test results available. Run comprehensive test first.")
            return
        
        # 创建可视化报告
        self.visualizer.create_final_report(self.test_results['all_results'])
        
        # 生成详细的episode分析
        for i, episode_result in enumerate(self.test_results['detailed_episodes']):
            episode_num = episode_result['episode_data'].get('episode', i * 10)
            self.visualizer.plot_episode_analysis(
                episode_result['episode_data'], episode_num
            )
        
        # 保存统计报告
        self._save_statistical_report()
        
        print(f"Test report generated in {self.results_dir}")
    
    def _save_statistical_report(self):
        """保存统计报告"""
        stats = self.test_results['overall_stats']
        
        # 创建文本报告
        report_text = f"""
=== Kyle Auction Model Test Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== Overall Performance ===
Mean Reward: {stats['mean_reward']:.4f}
Std Reward: {stats['std_reward']:.4f}
Sharpe Ratio: {stats['sharpe_ratio']:.4f}
Win Rate: {stats['win_rate']:.2%}

Total Profit: {stats['total_profit']:.4f}
Mean Profit per Episode: {stats['mean_profit']:.4f}
Profit Std: {stats['std_profit']:.4f}

Max Reward: {stats['max_reward']:.4f}
Min Reward: {stats['min_reward']:.4f}

=== Market Metrics ===
Mean Market Efficiency: {stats['mean_market_efficiency']:.4f}
Mean Volatility: {stats['mean_volatility']:.4f}
Consistency Score: {stats['consistency']:.4f}

=== Configuration ===
{json.dumps(self.config, indent=2)}
        """
        
        # 保存文本报告
        with open(os.path.join(self.results_dir, 'test_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # 保存JSON格式的详细数据
        with open(os.path.join(self.results_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            json_results = self._convert_for_json(self.test_results)
            json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    def _convert_for_json(self, obj):
        """转换对象为JSON可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def compare_strategies(self, strategy_configs: Dict[str, Dict]):
        """比较不同策略的性能"""
        comparison_results = {}
        
        for strategy_name, config in strategy_configs.items():
            print(f"Testing strategy: {strategy_name}")
            
            # 更新配置
            self.config.update(config)
            
            # 重新设置环境和智能体
            self.setup_environment()
            
            # 运行测试
            results = self.run_comprehensive_test(num_episodes=50)
            comparison_results[strategy_name] = results['all_results']
        
        # 创建比较图表
        self.visualizer.plot_strategy_comparison(comparison_results)
        
        return comparison_results

def load_test_config(config_path: str = None) -> Dict:
    """加载测试配置"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 默认测试配置
    return {
        'environment': {
            'num_auctions': 50,
            'num_traders': 5,
            'asset_value_mean': 100.0,
            'asset_value_std': 10.0,
            'noise_std': 1.0,
            'initial_price': 100.0,
            'max_position': 10.0,
            'transaction_cost': 0.01
        },
        'testing': {
            'num_episodes': 100,
            'record_frequency': 10
        }
    }

if __name__ == "__main__":
    # 示例测试流程
    config = load_test_config()
    
    # 创建测试器
    tester = ModelTester(config)
    
    # 设置环境
    tester.setup_environment()
    
    # 加载训练好的模型
    model_paths = {
        'insider': 'results/training/latest/insider_model.pth',
        'market_maker': 'results/training/latest/market_maker_model.pth'
    }
    tester.load_trained_agents(model_paths)
    
    # 运行综合测试
    results = tester.run_comprehensive_test(num_episodes=100)
    
    # 生成报告
    tester.generate_test_report()
    
    print("Testing completed!")
    print(f"Results saved in: {tester.results_dir}")