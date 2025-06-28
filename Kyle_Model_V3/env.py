import gymnasium as gym
import numpy as np

class EnhancedInsiderKyleEnv(gym.Env):
    def __init__(self, T=10, sigma_u=0.8, sigma_v=1.2, lambda_val=None, max_action=None, 
                 seed=None, dynamic_lambda=True, super_horizon=1):
        super().__init__()
        # 模型参数
        self.T = T                      # 内层轮数（单段信息的交易轮数）
        self.sigma_u = sigma_u          # 噪声交易的标准差
        self.sigma_v = sigma_v          # 真值先验标准差
        self.dynamic_lambda = dynamic_lambda  # 是否使用动态λ
        self.super_horizon = super_horizon    # 外层epoch数量（多少段信息）
        
        # ★ 若用户没写 λ 或动作上限→按理论值自动推
        if lambda_val is None:
            beta_1 = sigma_u / sigma_v * np.sqrt(self.T / (self.T + 1))
            lambda_val = 1.0 / (2.0 * beta_1)
        self.lambda_val = lambda_val    # 初始价格冲击系数（仅在dynamic_lambda=False时使用）
        
        if max_action is None:
            beta_1 = sigma_u / sigma_v * np.sqrt(self.T / (self.T + 1))
            self.max_action = 2.0 * beta_1 * sigma_v  # ≈ 2β₁σ_v
        else:
            self.max_action = max_action    # 动作空间上下限
        
        # 定义动作与观察空间
        self.action_space = gym.spaces.Box(low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.float32)
        # 观察包括: [规范化时间索引, 当前价格p_{t-1}, 真值v, 外层epoch进度]
        obs_low = np.array([0.0, -np.inf, -np.inf, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, np.inf, np.inf, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # 历史数据记录
        self.lambda_hist = []       # λₜ历史
        self.beta_hist = []         # βₜ历史
        self.price_impact_hist = [] # 价格冲击历史
        self.order_flow_hist = []   # 订单流历史
        self.noise_hist = []        # 噪声交易历史
        self.action_hist = []       # 内幕交易历史
        self.value_hist = []        # 真值历史（记录每段的v）
        self.segment_boundaries = [] # 记录每段的边界
        
        # 随机种子与内部状态初始化
        self.np_random = np.random.RandomState()
        if seed is not None:
            self.seed(seed)
        
        # 初始化状态
        self.outer_cnt = -1  # 外层epoch计数
        self.reset()
    
    def seed(self, seed=None):
        """设置随机种子以重现结果。"""
        self.np_random.seed(seed)
    
    def _theory_beta(self):
        """计算理论βₜ值（考虑剩余轮数）"""
        remain = self.T - self.t + 1  # 还剩多少内层步骤
        if remain <= 0:
            return 0.0
        return (self.sigma_u / np.sqrt(self.cur_var)) * np.sqrt(remain / (remain + 1))
    
    def _reset_inner_state(self):
        """重置内层状态（新的信息段）"""
        # 生成新的真实价值（每段独立）
        new_v = self.np_random.normal(loc=0.0, scale=self.sigma_v)
        self.v = new_v
        self.value_hist.append(new_v)
        
        # 价格保持连续，不在段切换时调整
        # 这符合Kyle模型理论：价格通过交易逐步发现价值
        
        # 重置内层计数和方差
        self.t = 0
        self.cur_var = self.sigma_v**2      # 方差重新拉满
        
        # ★ 重新计算理论 β₁、λ₁ 作为段首种子
        beta_1 = self.sigma_u / np.sqrt(self.cur_var) * np.sqrt(self.T / (self.T + 1))
        self.lambda_t = 1.0 / (2.0 * beta_1)
        
        # 记录段边界
        if len(self.segment_boundaries) == 0:
            self.segment_boundaries.append(0)
        else:
            self.segment_boundaries.append(len(self.lambda_hist))
    
    def reset(self, seed=None, options=None):
        """开始新的episode，初始化状态。"""
        if seed is not None:
            self.seed(seed)
        
        # 外层epoch计数
        self.outer_cnt += 1
        
        # 初始价格（如果是第一次重置，从0开始；否则保持连续性）
        if self.outer_cnt == 0:
            self.current_price = 0.0
        # 如果不是第一次重置，价格保持连续（不重置current_price）
        
        # 重置历史记录（仅在完全重新开始时）
        if self.outer_cnt == 0:
            self.lambda_hist = []
            self.beta_hist = []
            self.price_impact_hist = []
            self.order_flow_hist = []
            self.noise_hist = []
            self.action_hist = []
            self.value_hist = []
            self.segment_boundaries = []
        
        # 重置内层状态
        self._reset_inner_state()
        
        # 构造初始观测
        time_index = 0.0
        super_progress = self.outer_cnt / float(self.super_horizon)
        obs = np.array([time_index, self.current_price, self.v, super_progress], dtype=np.float32)
        return obs, {}
    
    def step(self, action):
        """执行智能体动作，推进市场到下一轮。"""
        self.t += 1
        x = float(action[0])  # 提交的订单量
        
        # 随机抽取本轮噪声交易量 u_t
        u = self.np_random.normal(loc=0.0, scale=self.sigma_u)
        
        # 计算总订单流 Q_t = x + u
        Q = x + u
        
        # 计算理论βₜ
        beta_star = self._theory_beta()
        self.beta_hist.append(beta_star)
        
        # 动态更新λₜ（如果启用）
        if self.dynamic_lambda and beta_star > 1e-8:
            self.lambda_t = 1.0 / (2.0 * beta_star)
        else:
            self.lambda_t = self.lambda_val  # 使用固定值
        
        self.lambda_hist.append(self.lambda_t)
        
        # 根据定价规则更新价格 p_t = p_{t-1} + λ * Q
        prev_price = self.current_price
        price_impact = self.lambda_t * Q
        self.current_price = prev_price + price_impact
        
        # 更新条件方差（精确贝叶斯更新）
        if beta_star > 1e-8:
            self.cur_var = (self.cur_var * self.sigma_u**2) / (beta_star**2 * self.cur_var + self.sigma_u**2)
        
        # 记录历史数据
        self.price_impact_hist.append(price_impact)
        self.order_flow_hist.append(Q)
        self.noise_hist.append(u)
        self.action_hist.append(x)
        
        # 计算奖励 r_t = x * (v - p_t)
        reward = x * (self.v - self.current_price)
        
        # 判断是否结束内层epoch
        done_inner = (self.t >= self.T)
        
        # 判断是否结束外层epoch（整个episode）
        done_outer = done_inner and (self.outer_cnt + 1 >= self.super_horizon)
        
        # 准备当前状态观测（使用当前段的真实价值）
        time_index = self.t / float(self.T)
        super_progress = (self.outer_cnt + 1) / float(self.super_horizon)
        obs = np.array([time_index, self.current_price, self.v, super_progress], dtype=np.float32)
        
        # 如果内层结束但外层未结束，切换到新的信息段
        if done_inner and not done_outer:
            # 保持价格连续性，重置内层状态
            self._reset_inner_state()
            # 外层计数增加
            self.outer_cnt += 1
        
        # info 字典包含调试信息
        info = {
            "noise": u,
            "Var(v|info)": self.cur_var,
            "beta_t": beta_star,
            "lambda_t": self.lambda_t,
            "price_impact": price_impact,
            "order_flow": Q,
            "inner_step": self.t,
            "outer_epoch": self.outer_cnt + 1,  # 从1开始计数
            "current_v": self.v,
            "segment_switch": done_inner and not done_outer,
            "segment_boundary": done_inner and not done_outer,
            "price": self.current_price,
            "value": self.v,
            "lambda": self.lambda_t,
            "conditional_var": self.cur_var
        }
        
        terminated = done_outer
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def get_theoretical_beta(self, t):
        """计算理论βₜ值（单期近似）"""
        if t == 0:
            return self.sigma_u / np.sqrt(self.cur_var)
        # 多期情况下βₜ应该递增
        remaining_var = self.cur_var
        return self.sigma_u / np.sqrt(remaining_var)
    
    def get_theoretical_lambda(self, beta_t):
        """计算理论λₜ值"""
        if abs(beta_t) > 1e-8:
            return 1.0 / (2.0 * beta_t)
        return self.lambda_val
    
    def get_market_depth(self):
        """计算市场深度（1/λ）"""
        if len(self.lambda_hist) > 0:
            return [1.0 / max(lam, 1e-8) for lam in self.lambda_hist]
        return []
    
    def get_information_leakage_rate(self):
        """计算信息泄露速率"""
        if len(self.lambda_hist) > 1:
            var_ratios = []
            for i in range(1, len(self.lambda_hist)):
                if self.cur_var > 1e-8:
                    var_ratios.append(-np.log(self.cur_var / self.sigma_v**2))
            return var_ratios
        return []
    
    def get_segment_info(self):
        """获取多段信息的统计"""
        return {
            'segment_boundaries': self.segment_boundaries,
            'value_hist': self.value_hist,
            'total_segments': len(self.value_hist),
            'current_segment': self.outer_cnt + 1
        }