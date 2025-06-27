import numpy as np
import yaml
import os
from stable_baselines3 import PPO
from env import EnhancedInsiderKyleEnv

def load_config(config_path="config.yaml"):
    """加载yaml配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"配置文件解析错误: {e}")
        return get_default_config()

def get_default_config():
    """获取默认配置（当yaml文件不存在时使用）"""
    return {
        'training_params': {
            'learning_rate': 0.0003,
            'n_steps': 3072,
            'batch_size': 64,
            'n_epochs': 10,
            'total_timesteps': 300000
        },
        'save_dir': './models',
        'configs': {
            'baseline_static': {
                'T': 10,
                'sigma_u': 0.8,
                'sigma_v': 1.2,
                'lambda_val': 0.3,
                'max_action': 3.0,
                'seed': 42,
                'dynamic_lambda': False,
                'super_horizon': 1
            }
        },
        'selected_configs': [],
        'options': {
            'verbose': True,
            'save_models': True,
            'parallel_training': False
        }
    }

def train_model(config_name, env_params, training_params, save_dir="./models", verbose=True):
    """训练单个配置的模型"""
    if verbose:
        print(f"\n开始训练配置: {config_name}")
        print(f"环境参数: {env_params}")
    
    # 创建环境实例
    env = EnhancedInsiderKyleEnv(**env_params)
    
    # 从训练参数中分离total_timesteps（使用copy避免修改原字典）
    training_params_copy = training_params.copy()
    total_timesteps = training_params_copy.pop('total_timesteps')
    
    # 创建PPO模型
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1 if verbose else 0,
        **training_params_copy
    )
    
    # 训练模型
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{config_name}_policy")
    model.save(model_path)
    
    if verbose:
        print(f"模型已保存为: {model_path}.zip")
    return model, env

def main(config_path="config.yaml"):
    """主训练函数 - 从yaml配置文件读取参数并训练模型"""
    
    # 加载配置
    config = load_config(config_path)
    
    # 提取配置参数
    training_params = config.get('training_params', {})
    save_dir = config.get('save_dir', './models')
    configs = config.get('configs', {})
    selected_configs = config.get('selected_configs', [])
    options = config.get('options', {})
    
    verbose = options.get('verbose', True)
    save_models = options.get('save_models', True)
    
    # 如果指定了特定配置，只训练这些配置
    if selected_configs:
        configs_to_train = {name: configs[name] for name in selected_configs if name in configs}
        if verbose:
            print(f"只训练指定的配置: {list(configs_to_train.keys())}")
    else:
        configs_to_train = configs
        if verbose:
            print(f"训练所有配置: {list(configs_to_train.keys())}")
    
    if not configs_to_train:
        print("错误：没有找到有效的配置")
        return {}
    
    trained_models = {}
    
    # 训练所有配置
    for config_name, env_params in configs_to_train.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"开始训练配置: {config_name}")
            if env_params.get('super_horizon', 1) > 1:
                print(f"多段信息配置 - 段数: {env_params['super_horizon']}")
            print(f"{'='*50}")
        
        try:
            model, env = train_model(
                config_name, 
                env_params, 
                training_params, 
                save_dir if save_models else None,
                verbose
            )
            trained_models[config_name] = {'model': model, 'env_params': env_params}
            
            if verbose:
                print(f"配置 {config_name} 训练完成")
        except Exception as e:
            print(f"配置 {config_name} 训练失败: {e}")
            continue
    
    if verbose:
        print("\n=== 所有模型训练完成 ===")
        print("可用的模型配置:")
        for config_name in trained_models.keys():
            print(f"  - {config_name}")
    
    return trained_models

def print_config_info(config_path="config.yaml"):
    """打印配置文件信息"""
    config = load_config(config_path)
    
    print("=== 当前配置信息 ===")
    print(f"训练参数: {config.get('training_params', {})}")
    print(f"保存目录: {config.get('save_dir', './models')}")
    print(f"可用配置: {list(config.get('configs', {}).keys())}")
    print(f"选择的配置: {config.get('selected_configs', [])}")
    print(f"选项: {config.get('options', {})}")
    print("=" * 30)

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数指定配置文件
    config_file = "config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    # 打印配置信息
    print_config_info(config_file)
    
    # 开始训练
    trained_models = main(config_file)
    
    if trained_models:
        print("\n训练脚本执行完成！")
        print("请运行 visualize.py 或 analysis.py 来分析结果。")
    else:
        print("\n训练失败，请检查配置文件。")