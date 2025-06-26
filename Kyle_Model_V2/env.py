import gymnasium as gym
import numpy as np

class EnhancedInsiderKyleEnv(gym.Env):
    def __init__(self, T=10, sigma_u=0.8, sigma_v=1.2, lambda_val=0.3, max_action=3.0, seed=None, dynamic_lambda=False):
        super().__init__()
        # 模型参数
        self.T = T                  # 总轮数
        self.sigma_u = sigma_u      # 噪声交易的标准差
        self.sigma_v = sigma_v      # 真值先验标准差
        self.lambda_val = lambda_val  # 做市商定价系数 λ（可动态）
        self.max_action = max_action  # 动作空间上下限
        self.dynamic_lambda = dynamic_lambda  # 是否使用动态λ
        
        # 定义动作与观察空间
        self.action_space = gym.spaces.Box(low=-max_action, high=max_action, shape=(1,), dtype=np.float32)
        # 观察包括: [规范化时间索引, 当前价格p_{t-1}, 真值v]
        obs_low = np.array([0.0, -np.inf, -np.inf], dtype=np.float32)
        obs_high = np.array([1.0, np.inf, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # 新增：记录历史数据用于分析
        self.lambda_hist = []       # λₜ历史
        self.beta_hist = []         # βₜ历史
        self.price_impact_hist = [] # 价格冲击历史
        self.order_flow_hist = []   # 订单流历史
        self.noise_hist = []        # 噪声交易历史
        self.action_hist = []       # 内幕交易历史
        
        # 随机种子与内部状态初始化
        self.np_random = np.random.RandomState()
        if seed is not None:
            self.seed(seed)
        self.reset()
    
    def seed(self, seed=None):
        """设置随机种子以重现结果。"""
        self.np_random.seed(seed)
    
    def reset(self, seed=None, options=None):
        """开始新的 episode，初始化状态。"""
        if seed is not None:
            self.seed(seed)
        # 抽取一个真实价值 v ~ N(0, σ_v^2)
        self.v = self.np_random.normal(loc=0.0, scale=self.sigma_v)
        # 初始价格 p0 设为先验均值 (0假设)
        self.current_price = 0.0
        # 初始条件方差 Var0
        self.current_var = self.sigma_v**2
        # 轮次计数重置
        self.current_step = 0
        
        # 重置历史记录
        self.lambda_hist = []
        self.beta_hist = []
        self.price_impact_hist = []
        self.order_flow_hist = []
        self.noise_hist = []
        self.action_hist = []
        
        # 构造初始观测: time_index=0, p0, v
        obs = np.array([0.0, self.current_price, self.v], dtype=np.float32)
        return obs, {}
    
    def step(self, action):
        """执行智能体动作，推进市场到下一轮。"""
        self.current_step += 1
        x = float(action[0])  # 提交的订单量
        
        # 随机抽取本轮噪声交易量 u_t
        u = self.np_random.normal(loc=0.0, scale=self.sigma_u)
        
        # 计算总订单流 Q_t = x + u
        Q = x + u
        
        # === ① 估算本轮"经验 βₜ" ===
        prev_price = self.current_price
        if abs(self.v - prev_price) > 1e-8:
            beta_hat = x / (self.v - prev_price)
        else:
            beta_hat = 0.0
        self.beta_hist.append(beta_hat)
        
        # === ② 动态更新 λₜ（如果启用） ===
        if self.dynamic_lambda and abs(beta_hat) > 1e-8:
            # 根据 Kyle 理论： λₜ = prior_var / (prior_var + σ_u²/βₜ²)
            self.lambda_val = self.current_var / (self.current_var + self.sigma_u**2 / max(beta_hat**2, 1e-8))
        
        # === ③ 储存 λₜ ===
        self.lambda_hist.append(self.lambda_val)
        
        # 根据定价规则更新价格 p_t = p_{t-1} + λ * Q
        price_impact = self.lambda_val * Q
        self.current_price = prev_price + price_impact
        
        # 记录历史数据
        self.price_impact_hist.append(price_impact)
        self.order_flow_hist.append(Q)
        self.noise_hist.append(u)
        self.action_hist.append(x)
        
        # 计算奖励 r_t = x * (v - p_t)
        reward = x * (self.v - self.current_price)
        
        # 更新条件方差 (改进的贝叶斯更新机制)
        # 使用更保守的方差更新，避免方差下降过快
        prior_var = self.current_var
        # 计算信息精度的增量，考虑订单流的信息含量
        info_precision = (self.lambda_val**2) / (self.lambda_val**2 * prior_var + self.sigma_u**2)
        # 使用衰减因子使方差更新更平滑
        decay_factor = 0.1  # 控制学习速度
        precision_update = decay_factor * info_precision
        # 更新方差：1/new_var = 1/old_var + precision_update
        new_precision = (1.0 / prior_var) + precision_update
        self.current_var = 1.0 / new_precision
        
        # 准备下一个状态观测
        done = (self.current_step >= self.T)
        # 归一化当前轮次索引用于观测（如用总轮数归一化）
        time_index = self.current_step / float(self.T)
        obs = np.array([time_index, self.current_price, self.v], dtype=np.float32)
        
        # info 字典包含调试信息：如本轮噪声u和当前条件方差
        info = {
            "noise": u, 
            "Var(v|info)": self.current_var,
            "beta_hat": beta_hat,
            "lambda_t": self.lambda_val,
            "price_impact": price_impact,
            "order_flow": Q
        }
        
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def get_theoretical_beta(self, t):
        """计算理论βₜ值（单期近似）"""
        if t == 0:
            return self.sigma_u / np.sqrt(self.current_var)
        # 多期情况下βₜ应该递增
        remaining_var = self.current_var
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
                if self.current_var > 1e-8:
                    var_ratios.append(-np.log(self.current_var / self.sigma_v**2))
            return var_ratios
        return []