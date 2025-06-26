# Kyle (1985) 内幕交易模型 - 强化学习实现 V3

基于Kyle (1985) 经典内幕交易模型的强化学习实现，使用PPO算法训练内幕交易者的最优策略。

## 🆕 V3版本新功能

- **多段信息支持**: 实现类似epoch套epoch的多段信息机制
- **连续价格衔接**: 不同信息段之间价格平滑过渡
- **段间分析**: 新增段间价格连续性和信息传递效率分析
- **可视化增强**: 图表中显示段边界标记

## 🌟 项目特色

- **实时Kyle指标记录**: 动态跟踪λ(价格冲击)、β(交易强度)等关键指标
- **批量回归分析**: 自动进行价格冲击回归和效率分析
- **理论对比验证**: 将RL结果与Kyle理论预测进行对比
- **多配置实验**: 支持不同噪声水平、时间长度的对比实验
- **论文级可视化**: 生成高质量的学术图表
- **多段信息机制**: 支持连续的多段信息价值变化

### 🎯 核心功能
- **实时记录Kyle指标**: λₜ (价格冲击系数)、βₜ (交易强度)
- **批量回归分析**: 横截面回归估计经验价格冲击
- **理论对比验证**: 实证结果与Kyle理论值对比
- **多配置实验**: 静态/动态λ、不同噪声水平、长期交易
- **论文级可视化**: 中文字体支持，高质量图表输出

### 📊 分析指标
- 价格冲击系数 λₜ 及其演化
- 交易强度 βₜ 随时间变化
- 市场深度 (1/λₜ) 分析
- 价格效率 R² 计算
- 信息泄露速率测量
- 收益分析和夏普比率

## 文件结构

```
enhanced_kyle_model/
├── env.py              # 增强版Kyle环境
├── train.py            # 多配置训练脚本
├── visualize.py        # 基础可视化
├── analysis.py         # 高级分析和回归
├── requirements.txt    # 依赖包列表
├── README.md          # 项目说明
├── models/            # 训练好的模型 (运行后生成)
├── plots/             # 基础图表 (运行后生成)
├── analysis_plots/    # 分析图表 (运行后生成)
├── __pycache__/       # Python缓存文件
└── README.md          # 项目文档
```

## 快速开始

### 1. 环境设置

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用conda
conda install numpy pandas matplotlib scipy scikit-learn
pip install stable-baselines3 gymnasium
```

### 2. 训练模型

```bash
# 训练所有配置 (约需20-30分钟)
python train.py
```

训练将生成7个不同配置的模型：

#### 基础配置
- `baseline_static`: 基础静态λ配置
- `baseline_dynamic`: 动态λ配置
- `high_noise`: 高噪声环境
- `low_noise`: 低噪声环境
- `long_term`: 长期交易(20轮)

#### 🆕 多段信息配置
- `multi_segment_3`: 3段信息配置
  - `super_horizon=3`: 包含3个连续的信息段
  - 每段内信息价值保持一致，段间平滑过渡
  
- `multi_segment_5`: 5段信息配置
  - `super_horizon=5`: 包含5个连续的信息段
  - 更复杂的多段信息演化模式

### 3. 基础可视化

```bash
# 生成基础图表
python visualize.py
```

输出图表：
- 价格路径与真实价值
- 条件方差演化
- 收益分析
- λₜ 和 βₜ 演化曲线
- 市场深度变化

### 4. 高级分析

```bash
# 批量回归分析 (约需10-15分钟)
python analysis.py
```

生成分析报告：
- 价格冲击回归: ΔP = λ × Q + ε
- 理论值对比验证
- 配置间性能比较
- 详细统计指标

## 配置说明

### 环境参数

| 参数 | 含义 | 基础值 | 说明 |
|------|------|--------|------|
| `T` | 内层交易轮数 | 10 | 每个信息段的交易回合数 |
| `sigma_u` | 噪声交易标准差 | 0.8 | 控制市场噪声水平 |
| `sigma_v` | 真值先验标准差 | 1.2 | 信息价值的不确定性 |
| `lambda_val` | 价格冲击系数 | 0.3 | 做市商定价敏感度 |
| `max_action` | 最大交易量 | 3.0 | 限制内幕交易规模 |
| `dynamic_lambda` | 动态λ开关 | False | 是否使用自适应定价 |
| 🆕 `super_horizon` | 信息段数 | 1 | 外层epoch数量，总轮数=T×super_horizon |

### 🆕 多段信息机制

多段信息是V3版本的核心新功能，实现了类似"epoch套epoch"的机制：

**工作原理**：
- **外层epoch**: `super_horizon`个信息段
- **内层epoch**: 每段包含`T`轮交易
- **总轮数**: `super_horizon × T`

**关键特性**：
1. **段内一致性**: 每个信息段内，真值`v`保持不变
2. **段间连续性**: 价格在段边界处平滑衔接，无跳跃
3. **信息更新**: 每个新段开始时生成新的真值
4. **价格连续**: 新段的起始价格 = 上一段的结束价格

**配置示例**：
```python
# 3段信息配置
config = {
    'T': 10,              # 每段10轮
    'super_horizon': 3,   # 3个信息段
    # 总共30轮交易
}
```

### 训练参数

| 参数 | V3值 | V2值 | 说明 |
|------|------|------|----- |
| `learning_rate` | 0.0003 | 0.0003 | PPO学习率 |
| `n_steps` | 3072 | 2048 | 每次更新的步数（增加以适应多段信息） |
| `batch_size` | 64 | 64 | 批处理大小 |
| `n_epochs` | 10 | 10 | 每次更新的训练轮数 |
| `total_timesteps` | 300000 | 200000 | 总训练步数（增加以适应更复杂的多段学习） |

**V3训练调整说明**：
- `n_steps`增加50%：多段信息需要更长的episode
- `total_timesteps`增加50%：多段配置需要更多训练时间
- 其他参数保持不变以确保兼容性

## 理论背景

### Kyle (1985) 模型核心

1. **价格形成**: pₜ = pₜ₋₁ + λₜ × Qₜ
2. **订单流**: Qₜ = xₜ + uₜ (内幕交易 + 噪声交易)
3. **最优策略**: xₜ = βₜ × (v - pₜ₋₁)
4. **理论关系**: λₜ = 1/(2βₜ)

### 关键指标

- **价格冲击 λₜ**: 单位订单流对价格的影响
- **交易强度 βₜ**: 内幕交易者对信息的利用程度
- **市场深度**: 1/λₜ，衡量市场流动性
- **信息效率**: 价格中包含的私人信息比例

## 实验结果解读

### 预期结果模式

1. **λₜ 递减**: 随着信息泄露，价格冲击应逐渐降低
2. **βₜ 递增**: 临近结束时交易强度增加
3. **方差递减**: 条件方差Var[v|pₜ]单调下降
4. **价格收敛**: 最终价格接近真实价值v

### 配置比较

- **高噪声 vs 低噪声**: 噪声越高，λ越小，市场深度越大
- **静态 vs 动态λ**: 动态λ能更好地反映信息结构变化
- **短期 vs 长期**: 更多轮次允许更完整的信息传递

## 扩展功能

### 自定义配置

```python
# 在train.py中添加新配置
custom_config = {
    'T': 15,
    'sigma_u': 1.0,
    'sigma_v': 2.0,
    'lambda_val': 0.25,
    'max_action': 4.0,
    'dynamic_lambda': True
}
```

### 高级分析

```python
# 在analysis.py中添加自定义指标
def custom_metric(step_df):
    # 计算自定义指标
    return result
```

## 故障排除

### 常见问题

1. **中文显示问题**: 
   - 项目已自动检测并配置中文字体
   - 支持的字体: SimHei, Heiti TC, STHeiti, PingFang SC, Microsoft YaHei等
   - 如果仍有问题，运行 `python font_config.py` 测试字体配置
   - macOS用户通常使用Heiti TC，Windows用户使用SimHei
2. **内存不足**: 减少analysis.py中的episodes数量
3. **训练不收敛**: 调整learning_rate或增加total_timesteps
4. **回归失败**: 检查数据中是否有足够的非零订单流

### 性能优化

- 使用GPU加速训练: `pip install torch[cuda]`
- 并行化分析: 修改analysis.py使用multiprocessing
- 减少可视化频率: 在visualize.py中采样显示

## V3版本使用指南

### 快速开始

```bash
# 1. 快速验证V3功能
python quick_test.py

# 2. 完整功能测试
python test_v3_features.py

# 3. 版本对比
python version_comparison.py

# 4. 运行完整实验
python run_experiment.py
```

### 多段信息配置示例

```python
# 三段信息配置
multi_segment_config = {
    'T': 5,                    # 每段5轮
    'sigma_u': 0.8,
    'sigma_v': 1.2,
    'lambda_val': 0.3,
    'max_action': 3.0,
    'dynamic_lambda': True,
    'super_horizon': 3         # 3段信息
}

# 创建环境
from env import EnhancedInsiderKyleEnv
env = EnhancedInsiderKyleEnv(**multi_segment_config)

# 运行episode
obs, info = env.reset()
while not done:
    action = agent.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    # 检查段信息
    current_segment = info['outer_epoch']
    is_boundary = info['segment_boundary']
    print(f"段{current_segment}, 边界: {is_boundary}")
```

### 多段信息分析

```python
# 使用专门的多段信息分析工具
from multi_segment_analysis import (
    analyze_segment_transitions,
    analyze_segment_efficiency,
    plot_multi_segment_analysis,
    generate_multi_segment_report
)

# 分析段间转换
transition_analysis = analyze_segment_transitions(data)
print(f"价格连续性得分: {transition_analysis['price_continuity_score']}")

# 分析段效率
efficiency_analysis = analyze_segment_efficiency(data)
for i, segment in enumerate(efficiency_analysis['segments']):
    print(f"段{i}: 信息效率={segment['info_efficiency']:.4f}")

# 生成完整报告
report = generate_multi_segment_report(data)
```

### V2兼容性

```python
# V3完全兼容V2，只需设置super_horizon=1
v2_compatible_config = {
    'T': 10,
    'sigma_u': 0.8,
    'sigma_v': 1.2,
    'lambda_val': 0.3,
    'max_action': 3.0,
    'dynamic_lambda': True,
    'super_horizon': 1  # V2兼容模式
}
```

### 高级功能

#### 自定义多段信息策略

```python
# 基于段信息的智能策略
class MultiSegmentStrategy:
    def __init__(self):
        self.segment_history = {}
    
    def predict(self, obs, info):
        current_segment = info['outer_epoch']
        
        # 根据段信息调整策略
        if current_segment not in self.segment_history:
            # 新段开始，重置策略参数
            self.segment_history[current_segment] = {
                'start_price': info['price'],
                'actions': []
            }
        
        # 段内策略逻辑
        segment_data = self.segment_history[current_segment]
        # ... 策略实现
        
        return action
```

#### 段间信息传递分析

```python
# 分析信息在段间的传递效率
def analyze_info_transfer(data):
    segments = data['segments']
    boundaries = data['segment_boundaries']
    
    transfer_efficiency = []
    
    for i, boundary_idx in enumerate(boundaries[:-1]):
        # 计算段间信息传递效率
        prev_segment_end = boundary_idx
        next_segment_start = boundary_idx + 1
        
        price_continuity = abs(
            data['prices'][next_segment_start] - 
            data['prices'][prev_segment_end]
        )
        
        transfer_efficiency.append(1 / (1 + price_continuity))
    
    return {
        'mean_transfer_efficiency': np.mean(transfer_efficiency),
        'transfer_scores': transfer_efficiency
    }
```

### 性能优化建议

1. **训练优化**:
   ```python
   # 多段信息训练建议参数
   training_params = {
       'learning_rate': 0.0003,
       'n_steps': 3072,        # 增加以适应更长episode
       'batch_size': 64,
       'n_epochs': 10,
       'total_timesteps': 300000  # 增加训练时间
   }
   ```

2. **内存优化**:
   ```python
   # 大规模多段信息实验
   large_scale_config = {
       'super_horizon': 10,  # 10段信息
       'T': 20,             # 每段20轮
       # 总轮数: 200轮
   }
   
   # 建议分批处理
   batch_size = 50  # 每批50个episode
   ```

3. **可视化优化**:
   ```python
   # 大数据量可视化
   plot_config = {
       'sample_rate': 10,    # 每10步采样一次
       'max_segments': 5,    # 最多显示5段
       'figure_size': (15, 10)
   }
   ```

## 引用

如果使用本项目，请引用：

```
Kyle, A. S. (1985). Continuous auctions and insider trading. 
Econometrica: Journal of the Econometric Society, 1315-1335.
```

## 许可证

MIT License - 详见LICENSE文件

## 贡献

欢迎提交Issue和Pull Request！

---

**注意**: 首次运行完整分析可能需要30-45分钟，建议先运行单个配置测试。