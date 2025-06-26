# 增强版Kyle模型强化学习项目

基于Kyle (1985) 内幕交易模型的强化学习实现，包含完整的理论指标分析和论文级结果输出。

## 项目特色

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

训练将生成5个不同配置的模型：
- `baseline_static`: 基础静态λ配置
- `baseline_dynamic`: 动态λ配置
- `high_noise`: 高噪声环境
- `low_noise`: 低噪声环境
- `long_term`: 长期交易(20轮)

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
| `T` | 交易轮数 | 10 | 总的交易回合数 |
| `sigma_u` | 噪声交易标准差 | 0.8 | 控制市场噪声水平 |
| `sigma_v` | 真值先验标准差 | 1.2 | 信息价值的不确定性 |
| `lambda_val` | 价格冲击系数 | 0.3 | 做市商定价敏感度 |
| `max_action` | 最大交易量 | 3.0 | 限制内幕交易规模 |
| `dynamic_lambda` | 动态λ开关 | False | 是否使用自适应定价 |

### 训练参数

| 参数 | 值 | 说明 |
|------|----|----- |
| `learning_rate` | 0.0003 | PPO学习率 |
| `n_steps` | 2048 | 每次更新的步数 |
| `batch_size` | 64 | 批处理大小 |
| `n_epochs` | 10 | 每次更新的训练轮数 |
| `total_timesteps` | 200000 | 总训练步数 |

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

1. **中文显示问题**: 确保安装了中文字体
2. **内存不足**: 减少analysis.py中的episodes数量
3. **训练不收敛**: 调整learning_rate或增加total_timesteps
4. **回归失败**: 检查数据中是否有足够的非零订单流

### 性能优化

- 使用GPU加速训练: `pip install torch[cuda]`
- 并行化分析: 修改analysis.py使用multiprocessing
- 减少可视化频率: 在visualize.py中采样显示

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