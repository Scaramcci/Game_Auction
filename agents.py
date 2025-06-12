import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from collections import deque
import random
from auction_environment import TraderType

class BaseAgent(ABC):
    """基础Agent抽象类"""
    
    def __init__(self, agent_id: int, trader_type: TraderType):
        self.agent_id = agent_id
        self.trader_type = trader_type
        self.action_history = []
        self.reward_history = []
        
    @abstractmethod
    def act(self, observation: np.ndarray, info: Dict = None) -> float:
        """根据观测选择动作"""
        pass
    
    @abstractmethod
    def update(self, observation: np.ndarray, action: float, reward: float, 
               next_observation: np.ndarray, done: bool, info: Dict = None):
        """更新Agent"""
        pass
    
    def reset(self):
        """重置Agent状态"""
        self.action_history = []
        self.reward_history = []

class DQNNetwork(nn.Module):
    """DQN网络"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class InsiderAgent(BaseAgent):
    """内幕交易者Agent - 使用DQN"""
    
    def __init__(self, agent_id: int, observation_dim: int, action_dim: int = 21,
                 learning_rate: float = 0.001, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 memory_size: int = 10000, batch_size: int = 32):
        
        super().__init__(agent_id, TraderType.INSIDER)
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim  # 离散化动作空间
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # 动作映射：将离散动作映射到连续交易量
        self.action_space = np.linspace(-50, 50, action_dim)
        
        # 神经网络
        self.q_network = DQNNetwork(observation_dim, action_dim)
        self.target_network = DQNNetwork(observation_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=memory_size)
        
        # 更新目标网络
        self.update_target_network()
        
        # 训练统计
        self.training_step = 0
        self.target_update_freq = 100
        
    def act(self, observation: np.ndarray, info: Dict = None) -> float:
        """ε-贪婪策略选择动作"""
        if np.random.random() < self.epsilon:
            # 随机动作
            action_idx = np.random.randint(self.action_dim)
        else:
            # 贪婪动作
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                q_values = self.q_network(obs_tensor)
                action_idx = q_values.argmax().item()
        
        # 映射到连续动作
        action = self.action_space[action_idx]
        self.action_history.append(action)
        
        return action
    
    def update(self, observation: np.ndarray, action: float, reward: float,
               next_observation: np.ndarray, done: bool, info: Dict = None):
        """更新DQN"""
        # 找到动作索引
        action_idx = np.argmin(np.abs(self.action_space - action))
        
        # 存储经验
        self.memory.append((observation, action_idx, reward, next_observation, done))
        self.reward_history.append(reward)
        
        # 训练
        if len(self.memory) >= self.batch_size:
            self._train()
        
        # 更新ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _train(self):
        """训练DQN"""
        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        
        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_observations = torch.FloatTensor(next_observations)
        dones = torch.BoolTensor(dones)
        
        # 当前Q值
        current_q_values = self.q_network(observations).gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_observations).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # 损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        # 更新目标网络
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']

class MarketMakerAgent(BaseAgent):
    """做市商Agent - 使用简单的均值回归策略"""
    
    def __init__(self, agent_id: int, learning_rate: float = 0.01):
        super().__init__(agent_id, TraderType.MARKET_MAKER)
        self.learning_rate = learning_rate
        self.price_memory = deque(maxlen=20)
        self.position_target = 0.0  # 目标持仓
        
    def act(self, observation: np.ndarray, info: Dict = None) -> float:
        """做市商策略：均值回归 + 库存管理"""
        current_price = observation[0]
        position = observation[11]  # 当前持仓
        
        self.price_memory.append(current_price)
        
        if len(self.price_memory) < 5:
            return 0.0
        
        # 计算价格均值
        price_mean = np.mean(list(self.price_memory))
        
        # 均值回归信号
        mean_reversion_signal = (price_mean - current_price) * 0.1
        
        # 库存管理信号
        inventory_signal = (self.position_target - position) * 0.05
        
        # 组合信号
        action = mean_reversion_signal + inventory_signal
        
        # 限制动作范围
        action = np.clip(action, -10, 10)
        
        self.action_history.append(action)
        return action
    
    def update(self, observation: np.ndarray, action: float, reward: float,
               next_observation: np.ndarray, done: bool, info: Dict = None):
        """更新做市商策略"""
        self.reward_history.append(reward)
        
        # 简单的适应性调整
        if len(self.reward_history) > 10:
            recent_rewards = self.reward_history[-10:]
            if np.mean(recent_rewards) < 0:
                # 如果最近表现不好，调整目标持仓
                self.position_target *= 0.95

class NoiseAgent(BaseAgent):
    """噪音交易者Agent - 随机交易"""
    
    def __init__(self, agent_id: int, noise_std: float = 5.0):
        super().__init__(agent_id, TraderType.NOISE)
        self.noise_std = noise_std
        
    def act(self, observation: np.ndarray, info: Dict = None) -> float:
        """随机交易"""
        action = np.random.normal(0, self.noise_std)
        action = np.clip(action, -20, 20)  # 限制范围
        self.action_history.append(action)
        return action
    
    def update(self, observation: np.ndarray, action: float, reward: float,
               next_observation: np.ndarray, done: bool, info: Dict = None):
        """噪音交易者不需要学习"""
        self.reward_history.append(reward)

class AdaptiveInsiderAgent(InsiderAgent):
    """自适应内幕交易者 - 改进版本"""
    
    def __init__(self, agent_id: int, observation_dim: int, **kwargs):
        super().__init__(agent_id, observation_dim, **kwargs)
        
        # 添加额外的策略参数
        self.risk_aversion = 0.1
        self.information_decay = 0.95
        self.market_impact_sensitivity = 0.1
        
    def act(self, observation: np.ndarray, info: Dict = None) -> float:
        """考虑风险和市场影响的改进策略"""
        current_price = observation[0]
        position = observation[11]
        true_value = observation[13] if len(observation) > 13 else current_price
        
        # 基础信号：价值差异
        value_signal = true_value - current_price
        
        # 风险调整
        risk_adjusted_signal = value_signal * (1 - self.risk_aversion * abs(position) / 50)
        
        # 市场影响考虑
        if info and 'lambda' in info:
            market_impact = info['lambda'] * abs(risk_adjusted_signal)
            risk_adjusted_signal *= (1 - self.market_impact_sensitivity * market_impact)
        
        # 使用DQN选择最终动作，但加入先验知识
        if np.random.random() < self.epsilon:
            # 探索时加入一些先验知识
            if abs(risk_adjusted_signal) > 5:
                action = np.sign(risk_adjusted_signal) * np.random.uniform(5, 15)
            else:
                action = np.random.uniform(-10, 10)
        else:
            # 利用时使用DQN
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                q_values = self.q_network(obs_tensor)
                action_idx = q_values.argmax().item()
                action = self.action_space[action_idx]
        
        self.action_history.append(action)
        return action

def create_agent(agent_type: str, agent_id: int, observation_dim: int, **kwargs) -> BaseAgent:
    """Agent工厂函数"""
    if agent_type == "insider":
        return InsiderAgent(agent_id, observation_dim, **kwargs)
    elif agent_type == "adaptive_insider":
        return AdaptiveInsiderAgent(agent_id, observation_dim, **kwargs)
    elif agent_type == "market_maker":
        return MarketMakerAgent(agent_id, **kwargs)
    elif agent_type == "noise":
        return NoiseAgent(agent_id, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")