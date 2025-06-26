import gymnasium as gym
import numpy as np

class InsiderKyleEnv(gym.Env):
    def __init__(self, T=10, sigma_u=1.0, sigma_v=1.0, lambda_val=0.5, max_action=5.0, seed=None):
        super().__init__()
        # 模型参数
        self.T = T                  # 总轮数
        self.sigma_u = sigma_u      # 噪声交易的标准差
        self.sigma_v = sigma_v      # 真值先验标准差
        self.lambda_val = lambda_val  # 做市商定价系数 λ（固定）
        self.max_action = max_action  # 动作空间上下限
        
        # 定义动作与观察空间
        self.action_space = gym.spaces.Box(low=-max_action, high=max_action, shape=(1,), dtype=np.float32)
        # 观察包括: [规范化时间索引, 当前价格p_{t-1}, 真值v]
        obs_low = np.array([0.0, -np.inf, -np.inf], dtype=np.float32)
        obs_high = np.array([1.0, np.inf, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
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
        # 根据定价规则更新价格 p_t = p_{t-1} + λ * Q
        prev_price = self.current_price
        self.current_price = prev_price + self.lambda_val * Q
        # 计算奖励 r_t = x * (v - p_t)
        reward = x * (self.v - self.current_price)
        # 更新条件方差 (可选简单近似：按贝叶斯公式或固定衰减因子)
        # 这里采用贝叶斯更新假设内幕策略线性，以估计方差减少
        prior_var = self.current_var
        # 计算协方差和方差用于更新
        cov_v_Q = prior_var  # 假设x≈线性v使Cov(v,Q)≈Var(v)
        var_Q = (cov_v_Q * self.lambda_val)**2 + self.sigma_u**2  # 近似 Var(Q) = (λVar(v))^2 + Var(u)
        self.current_var = prior_var - (cov_v_Q**2) / var_Q
        # 准备下一个状态观测
        done = (self.current_step >= self.T)
        # 归一化当前轮次索引用于观测（如用总轮数归一化）
        time_index = self.current_step / float(self.T)
        obs = np.array([time_index, self.current_price, self.v], dtype=np.float32)
        # info 字典包含调试信息：如本轮噪声u和当前条件方差
        info = {"noise": u, "Var(v|info)": self.current_var}
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info