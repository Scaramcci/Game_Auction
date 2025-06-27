# Kyle Model V3 配置系统使用说明

## 概述

Kyle Model V3 现在支持通过 YAML 配置文件来管理所有训练参数，使得参数调整更加方便和灵活。

## 配置文件结构

### config.yaml 文件包含以下主要部分：

1. **training_params**: PPO训练算法的参数
2. **save_dir**: 模型保存目录
3. **configs**: 各种环境配置
4. **selected_configs**: 指定要训练的配置（可选）
5. **options**: 训练选项

## 使用方法

### 1. 基本使用

```bash
# 使用默认配置文件 config.yaml
python train.py

# 使用自定义配置文件
python train.py my_config.yaml
```

### 2. 配置参数说明

#### 训练参数 (training_params)
- `learning_rate`: 学习率
- `n_steps`: 每次更新的步数
- `batch_size`: 批次大小
- `n_epochs`: 训练轮数
- `total_timesteps`: 总训练步数

#### 环境参数 (configs)
每个配置包含以下参数：
- `T`: 交易轮数
- `sigma_u`: 噪声标准差
- `sigma_v`: 信息价值标准差
- `lambda_val`: 价格冲击参数
- `max_action`: 最大交易量
- `seed`: 随机种子
- `dynamic_lambda`: 是否使用动态λ
- `super_horizon`: 信息段数（1为单段，>1为多段）

#### 训练选项 (options)
- `verbose`: 是否显示详细信息
- `save_models`: 是否保存训练好的模型
- `parallel_training`: 是否并行训练（暂未实现）

### 3. 选择性训练

如果只想训练特定配置，可以在 `selected_configs` 中指定：

```yaml
selected_configs: ["baseline_static", "multi_segment_3"]
```

如果 `selected_configs` 为空列表，则训练所有配置。

### 4. 自定义配置

你可以复制 `config.yaml` 并修改参数来创建自己的配置：

```bash
cp config.yaml my_experiment.yaml
# 编辑 my_experiment.yaml
python train.py my_experiment.yaml
```

### 5. 查看配置信息

运行训练脚本时，会自动显示当前配置信息：

```
=== 当前配置信息 ===
训练参数: {'learning_rate': 0.0003, ...}
保存目录: ./models
可用配置: ['baseline_static', 'baseline_dynamic', ...]
选择的配置: []
选项: {'verbose': True, 'save_models': True, ...}
==============================
```

## 预定义配置说明

1. **baseline_static**: 基础静态λ配置
2. **baseline_dynamic**: 基础动态λ配置
3. **high_noise**: 高噪声环境
4. **low_noise**: 低噪声环境
5. **long_term**: 长期交易（20轮）
6. **multi_segment_3**: 3段信息配置
7. **multi_segment_5**: 5段信息配置

## 故障排除

### 配置文件不存在
如果指定的配置文件不存在，程序会自动使用默认配置并显示警告。

### YAML 语法错误
如果配置文件有语法错误，程序会显示错误信息并使用默认配置。

### 缺少依赖
确保安装了 PyYAML：
```bash
pip install PyYAML>=6.0.0
```

或者安装所有依赖：
```bash
pip install -r requirements.txt
```

## 示例：快速实验

创建一个快速测试配置：

```yaml
# quick_test.yaml
training_params:
  total_timesteps: 50000  # 减少训练时间
  
configs:
  quick_test:
    T: 5
    sigma_u: 0.8
    sigma_v: 1.2
    lambda_val: 0.3
    max_action: 3.0
    seed: 42
    dynamic_lambda: true
    super_horizon: 1
    
selected_configs: ["quick_test"]
```

然后运行：
```bash
python train.py quick_test.yaml
```

这样可以快速测试代码是否正常工作。