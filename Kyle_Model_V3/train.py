import numpy as np
from stable_baselines3 import PPO
from env import EnhancedInsiderKyleEnv
import os

def train_model(config_name, env_params, training_params, save_dir="./models"):
    """训练单个配置的模型"""
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
        verbose=1,
        **training_params_copy
    )
    
    # 训练模型
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{config_name}_policy")
    model.save(model_path)
    
    print(f"模型已保存为: {model_path}.zip")
    return model, env

def main():
    """主训练函数 - 训练多种配置"""
    
    # 基础训练参数（针对多段信息调整）
    base_training_params = {
        'learning_rate': 0.0003,
        'n_steps': 3072,  # 增加以适应更长的episode
        'batch_size': 64,
        'n_epochs': 10,
        'total_timesteps': 300000  # 增加训练时间
    }
    
    # 配置1: 基础配置（静态λ）
    config1 = {
        'T': 10,
        'sigma_u': 0.8,
        'sigma_v': 1.2,
        'lambda_val': 0.3,
        'max_action': 3.0,
        'seed': 42,
        'dynamic_lambda': False,
        'super_horizon': 1  # 单段信息
    }
    
    # 配置2: 动态λ配置
    config2 = {
        'T': 10,
        'sigma_u': 0.8,
        'sigma_v': 1.2,
        'lambda_val': 0.3,  # 初始值
        'max_action': 3.0,
        'seed': 42,
        'dynamic_lambda': True,
        'super_horizon': 1  # 单段信息
    }
    
    # 配置3: 高噪声环境
    config3 = {
        'T': 10,
        'sigma_u': 1.5,  # 更高噪声
        'sigma_v': 1.2,
        'lambda_val': 0.2,  # 更低价格冲击
        'max_action': 3.0,
        'seed': 42,
        'dynamic_lambda': True,  # 改为动态
        'super_horizon': 1
    }
    
    # 配置4: 低噪声环境
    config4 = {
        'T': 10,
        'sigma_u': 0.5,  # 更低噪声
        'sigma_v': 1.2,
        'lambda_val': 0.4,  # 更高价格冲击
        'max_action': 3.0,
        'seed': 42,
        'dynamic_lambda': True,  # 改为动态
        'super_horizon': 1
    }
    
    # 配置5: 长期交易
    config5 = {
        'T': 20,  # 更多轮次
        'sigma_u': 0.8,
        'sigma_v': 1.5,  # 更高信息价值
        'lambda_val': 0.25,
        'max_action': 2.5,
        'seed': 42,
        'dynamic_lambda': True,
        'super_horizon': 1
    }
    
    # 配置6: 多段信息（3段）
    config6 = {
        'T': 10,
        'sigma_u': 0.8,
        'sigma_v': 1.2,
        'lambda_val': 0.3,
        'max_action': 3.0,
        'seed': 42,
        'dynamic_lambda': True,
        'super_horizon': 3  # 3段信息
    }
    
    # 配置7: 多段信息（5段）
    config7 = {
        'T': 10,
        'sigma_u': 0.8,
        'sigma_v': 1.2,
        'lambda_val': 0.3,
        'max_action': 3.0,
        'seed': 42,
        'dynamic_lambda': True,
        'super_horizon': 5  # 5段信息
    }
    
    configs = {
        'baseline_static': config1,
        'baseline_dynamic': config2,
        'high_noise': config3,
        'low_noise': config4,
        'long_term': config5,
        'multi_segment_3': config6,
        'multi_segment_5': config7
    }
    
    trained_models = {}
    
    # 训练所有配置
    for config_name, env_params in configs.items():
        print(f"\n{'='*50}")
        print(f"开始训练配置: {config_name}")
        if env_params.get('super_horizon', 1) > 1:
            print(f"多段信息配置 - 段数: {env_params['super_horizon']}")
        print(f"{'='*50}")
        
        model, env = train_model(config_name, env_params, base_training_params)
        trained_models[config_name] = {'model': model, 'env_params': env_params}
        
        print(f"配置 {config_name} 训练完成")
    
    print("\n=== 所有模型训练完成 ===")
    print("可用的模型配置:")
    for config_name in trained_models.keys():
        print(f"  - {config_name}")
    
    return trained_models

if __name__ == "__main__":
    trained_models = main()
    print("\n训练脚本执行完成！")
    print("请运行 visualize.py 或 analysis.py 来分析结果。")