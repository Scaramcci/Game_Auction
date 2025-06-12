import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import os
from datetime import datetime
import json
from tqdm import tqdm

from auction_environment import KyleAuctionEnvironment
from agents import create_agent, BaseAgent
from visualization import TrainingVisualizer

class TrainingManager:
    """训练管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.env = None
        self.agents = {}
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'profits': [],
            'positions': [],
            'prices': [],
            'order_flows': [],
            'market_efficiency': [],
            'volatility': []
        }
        
        # 创建结果目录
        self.results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def setup_environment(self):
        """设置环境"""
        env_config = self.config['environment']
        self.env = KyleAuctionEnvironment(**env_config)
        print(f"Environment created with {env_config['n_auctions']} auctions")
    
    def setup_agents(self):
        """设置智能体"""
        agents_config = self.config['agents']
        observation_dim = self.env.observation_space.shape[0]
        
        for agent_name, agent_config in agents_config.items():
            agent_type = agent_config['type']
            agent_id = agent_config['id']
            
            # 移除type和id，传递其余参数
            agent_params = {k: v for k, v in agent_config.items() if k not in ['type', 'id']}
            
            agent = create_agent(agent_type, agent_id, observation_dim, **agent_params)
            self.agents[agent_name] = agent
            print(f"Created {agent_type} agent: {agent_name}")
    
    def train_episode(self, episode: int) -> Dict:
        """训练一个episode"""
        obs = self.env.reset()
        done = False
        episode_data = {
            'rewards': [],
            'actions': [],
            'prices': [],
            'positions': [],
            'order_flows': [],
            'market_states': []
        }
        
        step = 0
        while not done and step < self.config['training']['max_steps_per_episode']:
            # 获取主要智能体（内幕交易者）的动作
            main_agent = self.agents[self.config['training']['main_agent']]
            action = main_agent.act(obs)
            
            # 执行动作
            next_obs, reward, done, info = self.env.step([action])
            
            # 更新智能体
            main_agent.update(obs, action, reward, next_obs, done, info)
            
            # 记录数据
            episode_data['rewards'].append(reward)
            episode_data['actions'].append(action)
            episode_data['prices'].append(info['current_price'])
            episode_data['positions'].append(info['insider_position'])
            episode_data['order_flows'].append(info['total_order_flow'])
            episode_data['market_states'].append(self.env.get_market_state())
            
            obs = next_obs
            step += 1
        
        # 计算episode指标
        total_reward = sum(episode_data['rewards'])
        avg_position = np.mean(episode_data['positions']) if episode_data['positions'] else 0
        price_volatility = np.std(episode_data['prices']) if len(episode_data['prices']) > 1 else 0
        
        # 计算市场效率（价格与真实价值的相关性）
        if len(episode_data['prices']) > 1:
            true_values = [state['asset_true_value'] for state in episode_data['market_states']]
            prices = episode_data['prices']
            try:
                market_efficiency = np.corrcoef(prices, true_values)[0, 1]
            except:
                market_efficiency = 0
        else:
            market_efficiency = 0
        
        # 计算总利润
        total_profit = episode_data['market_states'][-1]['traders'][0]['profit'] if episode_data['market_states'] else 0
        
        # 返回结果
        result = {
            'total_reward': total_reward,
            'metrics': {
                'total_profit': total_profit,
                'avg_position': avg_position,
                'price_volatility': price_volatility,
                'market_efficiency': market_efficiency
            },
            'data': episode_data
        }
        
        return result
    
    def evaluate_episode(self, episode: int) -> Dict:
        """评估一个episode（不更新智能体）"""
        obs = self.env.reset()
        done = False
        episode_data = {
            'rewards': [],
            'actions': [],
            'prices': [],
            'positions': [],
            'order_flows': [],
            'market_states': []
        }
        
        # 临时禁用探索
        main_agent = self.agents[self.config['training']['main_agent']]
        original_epsilon = getattr(main_agent, 'epsilon', 0)
        if hasattr(main_agent, 'epsilon'):
            main_agent.epsilon = 0
        
        step = 0
        while not done and step < self.config['training']['max_steps_per_episode']:
            action = main_agent.act(obs)
            next_obs, reward, done, info = self.env.step([action])
            
            episode_data['rewards'].append(reward)
            episode_data['actions'].append(action)
            episode_data['prices'].append(info['current_price'])
            episode_data['positions'].append(info['insider_position'])
            episode_data['order_flows'].append(info['total_order_flow'])
            episode_data['market_states'].append(self.env.get_market_state())
            
            obs = next_obs
            step += 1
        
        # 恢复探索率
        if hasattr(main_agent, 'epsilon'):
            main_agent.epsilon = original_epsilon
        
        # 计算评估指标
        avg_reward = np.mean(episode_data['rewards']) if episode_data['rewards'] else 0
        
        # 计算胜率（正收益的比例）
        positive_rewards = [r for r in episode_data['rewards'] if r > 0]
        win_rate = len(positive_rewards) / len(episode_data['rewards']) if episode_data['rewards'] else 0
        
        # 计算其他指标
        avg_position = np.mean(episode_data['positions']) if episode_data['positions'] else 0
        price_volatility = np.std(episode_data['prices']) if len(episode_data['prices']) > 1 else 0
        
        # 计算市场效率
        if len(episode_data['prices']) > 1:
            true_values = [state['asset_true_value'] for state in episode_data['market_states']]
            prices = episode_data['prices']
            try:
                market_efficiency = np.corrcoef(prices, true_values)[0, 1]
            except:
                market_efficiency = 0
        else:
            market_efficiency = 0
        
        # 计算总利润
        total_profit = episode_data['market_states'][-1]['traders'][0]['profit'] if episode_data['market_states'] else 0
        
        # 返回结果
        result = {
            'avg_reward': avg_reward,
            'win_rate': win_rate,
            'metrics': {
                'total_profit': total_profit,
                'avg_position': avg_position,
                'price_volatility': price_volatility,
                'market_efficiency': market_efficiency
            },
            'data': episode_data
        }
        
        return result
    
    def calculate_metrics(self, episode_data: Dict) -> Dict:
        """计算性能指标"""
        prices = np.array(episode_data['prices'])
        rewards = np.array(episode_data['rewards'])
        positions = np.array(episode_data['positions'])
        order_flows = np.array(episode_data['order_flows'])
        
        metrics = {
            'total_reward': np.sum(rewards),
            'final_profit': rewards[-1] if len(rewards) > 0 else 0,
            'avg_position': np.mean(np.abs(positions)),
            'max_position': np.max(np.abs(positions)),
            'price_volatility': np.std(prices) if len(prices) > 1 else 0,
            'order_flow_volatility': np.std(order_flows) if len(order_flows) > 1 else 0,
            'market_efficiency': self._calculate_market_efficiency(episode_data),
            'sharpe_ratio': self._calculate_sharpe_ratio(rewards)
        }
        
        return metrics
    
    def _calculate_market_efficiency(self, episode_data: Dict) -> float:
        """计算市场效率（价格与真实价值的接近程度）"""
        if not episode_data['market_states']:
            return 0.0
        
        price_errors = []
        for state in episode_data['market_states']:
            true_value = state.get('asset_true_value', 0)
            current_price = state.get('current_price', 0)
            if true_value != 0:
                error = abs(current_price - true_value) / true_value
                price_errors.append(error)
        
        return 1.0 - np.mean(price_errors) if price_errors else 0.0
    
    def _calculate_sharpe_ratio(self, rewards: np.ndarray) -> float:
        """计算夏普比率"""
        if len(rewards) < 2:
            return 0.0
        
        returns = np.diff(rewards)
        if np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns)
    
    def train(self):
        """主训练循环"""
        print("Starting training...")
        
        training_config = self.config['training']
        n_episodes = training_config['n_episodes']
        eval_frequency = training_config.get('eval_frequency', 100)
        save_frequency = training_config.get('save_frequency', 500)
        
        # 创建可视化器
        visualizer = TrainingVisualizer(self.results_dir)
        
        for episode in tqdm(range(n_episodes), desc="Training"):
            # 训练episode
            episode_data = self.train_episode(episode)
            metrics = self.calculate_metrics(episode_data)
            
            # 记录训练历史
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(metrics['total_reward'])
            self.training_history['profits'].append(metrics['final_profit'])
            self.training_history['positions'].append(metrics['avg_position'])
            self.training_history['prices'].append(episode_data['prices'])
            self.training_history['order_flows'].append(episode_data['order_flows'])
            self.training_history['market_efficiency'].append(metrics['market_efficiency'])
            self.training_history['volatility'].append(metrics['price_volatility'])
            
            # 定期评估
            if episode % eval_frequency == 0:
                eval_data = self.evaluate_episode(episode)
                eval_metrics = self.calculate_metrics(eval_data)
                
                print(f"Episode {episode}:")
                print(f"  Training Reward: {metrics['total_reward']:.2f}")
                print(f"  Eval Reward: {eval_metrics['total_reward']:.2f}")
                print(f"  Market Efficiency: {eval_metrics['market_efficiency']:.3f}")
                print(f"  Price Volatility: {eval_metrics['price_volatility']:.2f}")
                
                # 更新可视化
                visualizer.update_training_plots(self.training_history, episode)
                visualizer.plot_episode_analysis(eval_data, episode)
            
            # 定期保存模型
            if episode % save_frequency == 0 and episode > 0:
                self.save_models(episode)
        
        print("Training completed!")
        
        # 最终评估和可视化
        self.final_evaluation()
        visualizer.create_final_report(self.training_history)
    
    def save_models(self, episode: int):
        """保存模型"""
        models_dir = os.path.join(self.results_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'save_model'):
                model_path = os.path.join(models_dir, f'{agent_name}_episode_{episode}.pth')
                agent.save_model(model_path)
        
        print(f"Models saved at episode {episode}")
    
    def final_evaluation(self):
        """最终评估"""
        print("\nFinal Evaluation:")
        
        # 运行多个评估episode
        eval_results = []
        for i in range(10):
            eval_result = self.evaluate_episode(f"final_eval_{i}")
            # 从新的返回格式中提取数据
            eval_data = eval_result['data']
            # 使用新的返回格式中的metrics
            metrics = eval_result['metrics']
            # 添加其他指标
            metrics['total_reward'] = eval_result['avg_reward'] * len(eval_data['rewards'])
            metrics['win_rate'] = eval_result['win_rate']
            eval_results.append(metrics)
        
        # 计算平均性能
        avg_metrics = {}
        for key in eval_results[0].keys():
            avg_metrics[f'avg_{key}'] = np.mean([result[key] for result in eval_results])
            avg_metrics[f'std_{key}'] = np.std([result[key] for result in eval_results])
        
        # 保存评估结果
        with open(os.path.join(self.results_dir, 'final_evaluation.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        
        print(f"Average Total Reward: {avg_metrics['avg_total_reward']:.2f} ± {avg_metrics['std_total_reward']:.2f}")
        print(f"Average Market Efficiency: {avg_metrics['avg_market_efficiency']:.3f} ± {avg_metrics['std_market_efficiency']:.3f}")
        print(f"Average Sharpe Ratio: {avg_metrics.get('avg_sharpe_ratio', 0):.3f} ± {avg_metrics.get('std_sharpe_ratio', 0):.3f}")
        
        return avg_metrics

def load_config(config_path: str = None) -> Dict:
    """加载配置文件"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # 默认配置
    return {
        'environment': {
            'n_auctions': 100,
            'n_traders': 5,
            'asset_true_value_mean': 100.0,
            'asset_true_value_std': 10.0,
            'noise_std': 5.0,
            'initial_price': 100.0,
            'max_position': 50.0,
            'transaction_cost': 0.01
        },
        'agents': {
            'insider': {
                'type': 'adaptive_insider',
                'id': 0,
                'learning_rate': 0.001,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'memory_size': 10000,
                'batch_size': 32
            }
        },
        'training': {
            'n_episodes': 2000,
            'max_steps_per_episode': 100,
            'eval_frequency': 100,
            'save_frequency': 500,
            'main_agent': 'insider'
        }
    }

if __name__ == "__main__":
    # 加载配置
    config = load_config()
    
    # 创建训练管理器
    trainer = TrainingManager(config)
    
    # 设置环境和智能体
    trainer.setup_environment()
    trainer.setup_agents()
    
    # 开始训练
    trainer.train()