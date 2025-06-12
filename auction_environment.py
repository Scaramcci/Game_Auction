import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

class TraderType(Enum):
    INSIDER = "insider"
    MARKET_MAKER = "market_maker"
    NOISE = "noise"

@dataclass
class TraderState:
    """交易者状态"""
    trader_id: int
    trader_type: TraderType
    position: float = 0.0  # 持仓
    cash: float = 1000.0   # 现金
    profit: float = 0.0    # 累计利润
    private_value: Optional[float] = None  # 私有信息（仅内幕交易者有）
    
class KyleAuctionEnvironment(gym.Env):
    """Kyle模型拍卖环境"""
    
    def __init__(self, 
                 n_auctions: int = 100,
                 n_traders: int = 5,
                 asset_true_value_mean: float = 100.0,
                 asset_true_value_std: float = 10.0,
                 noise_std: float = 5.0,
                 initial_price: float = 100.0,
                 max_position: float = 100.0,
                 transaction_cost: float = 0.01):
        
        super().__init__()
        
        # 环境参数
        self.n_auctions = n_auctions
        self.n_traders = n_traders
        self.asset_true_value_mean = asset_true_value_mean
        self.asset_true_value_std = asset_true_value_std
        self.noise_std = noise_std
        self.initial_price = initial_price
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        
        # 状态变量
        self.current_auction = 0
        self.asset_true_value = None
        self.current_price = initial_price
        self.price_history = []
        self.order_flow_history = []
        self.traders = []
        
        # 动作空间：连续的交易量 [-max_position, max_position]
        self.action_space = spaces.Box(
            low=-max_position, 
            high=max_position, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # 观测空间：[当前价格, 历史价格(最近10期), 自己的持仓, 现金, 累计利润, 资产真实价值(仅内幕交易者)]
        obs_dim = 1 + 10 + 3 + 1  # 当前价格 + 历史价格 + 个人状态 + 私有信息
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_auction = 0
        
        # 生成资产真实价值
        self.asset_true_value = np.random.normal(
            self.asset_true_value_mean, 
            self.asset_true_value_std
        )
        
        self.current_price = self.initial_price
        self.price_history = [self.initial_price] * 10
        self.order_flow_history = []
        
        # 初始化交易者
        self.traders = []
        for i in range(self.n_traders):
            if i == 0:  # 第一个是内幕交易者
                trader_type = TraderType.INSIDER
                private_value = self.asset_true_value
            elif i == 1:  # 第二个是做市商
                trader_type = TraderType.MARKET_MAKER
                private_value = None
            else:  # 其余是噪音交易者
                trader_type = TraderType.NOISE
                private_value = None
                
            trader = TraderState(
                trader_id=i,
                trader_type=trader_type,
                private_value=private_value
            )
            self.traders.append(trader)
        
        return self._get_observation(0)  # 返回内幕交易者的观测
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步"""
        # action是内幕交易者的动作
        insider_order = float(action[0])
        
        # 生成其他交易者的订单
        orders = [insider_order]  # 内幕交易者
        
        # 做市商订单（简单策略：反向交易）
        market_maker_order = -0.1 * insider_order + np.random.normal(0, 1)
        orders.append(market_maker_order)
        
        # 噪音交易者订单
        for i in range(2, self.n_traders):
            noise_order = np.random.normal(0, self.noise_std)
            orders.append(noise_order)
        
        # 计算总订单流
        total_order_flow = sum(orders)
        self.order_flow_history.append(total_order_flow)
        
        # 更新价格（Kyle模型的线性价格影响）
        lambda_t = self._calculate_lambda()
        price_impact = lambda_t * total_order_flow
        new_price = self.current_price + price_impact
        
        # 更新交易者状态
        for i, order in enumerate(orders):
            trader = self.traders[i]
            
            # 更新持仓
            trader.position += order
            
            # 更新现金（考虑交易成本）
            cost = order * new_price + abs(order) * self.transaction_cost
            trader.cash -= cost
            
            # 计算即时利润
            if trader.trader_type == TraderType.INSIDER:
                # 内幕交易者知道真实价值
                instant_profit = order * (self.asset_true_value - new_price)
            else:
                # 其他交易者基于当前价格
                instant_profit = order * (self.current_price - new_price)
            
            trader.profit += instant_profit
        
        self.current_price = new_price
        self.price_history.append(new_price)
        if len(self.price_history) > 10:
            self.price_history.pop(0)
        
        self.current_auction += 1
        
        # 计算奖励（内幕交易者的利润）
        reward = self.traders[0].profit
        
        # 检查是否结束
        done = self.current_auction >= self.n_auctions
        
        # 额外信息
        info = {
            'current_price': self.current_price,
            'asset_true_value': self.asset_true_value,
            'total_order_flow': total_order_flow,
            'lambda': lambda_t,
            'insider_position': self.traders[0].position,
            'insider_profit': self.traders[0].profit
        }
        
        return self._get_observation(0), reward, done, info
    
    def _calculate_lambda(self) -> float:
        """计算价格影响系数λ"""
        # Kyle模型中的λ = σ_v / (σ_u * sqrt(T-t))
        remaining_time = max(1, self.n_auctions - self.current_auction)
        lambda_t = self.asset_true_value_std / (self.noise_std * np.sqrt(remaining_time))
        return lambda_t
    
    def _get_observation(self, trader_id: int) -> np.ndarray:
        """获取指定交易者的观测"""
        trader = self.traders[trader_id]
        
        obs = []
        
        # 当前价格
        obs.append(self.current_price)
        
        # 历史价格（最近10期）
        obs.extend(self.price_history)
        
        # 个人状态
        obs.append(trader.position)
        obs.append(trader.cash)
        obs.append(trader.profit)
        
        # 如果是内幕交易者，添加私有信息
        if trader.trader_type == TraderType.INSIDER:
            obs.append(self.asset_true_value)
        else:
            obs.append(0.0)  # 其他交易者没有私有信息
        
        return np.array(obs, dtype=np.float32)
    
    def get_market_state(self) -> Dict:
        """获取市场状态信息"""
        return {
            'current_auction': self.current_auction,
            'current_price': self.current_price,
            'asset_true_value': self.asset_true_value,
            'price_history': self.price_history.copy(),
            'order_flow_history': self.order_flow_history.copy(),
            'traders': [{
                'id': t.trader_id,
                'type': t.trader_type.value,
                'position': t.position,
                'cash': t.cash,
                'profit': t.profit
            } for t in self.traders]
        }
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"Auction {self.current_auction}/{self.n_auctions}")
            print(f"Current Price: {self.current_price:.2f}")
            print(f"True Value: {self.asset_true_value:.2f}")
            print("Traders:")
            for trader in self.traders:
                print(f"  {trader.trader_type.value}: Position={trader.position:.2f}, "
                      f"Cash={trader.cash:.2f}, Profit={trader.profit:.2f}")
            print("-" * 50)