import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def update_training_plots(self, training_history: Dict, current_episode: int):
        """更新训练过程图表"""
        if current_episode % 100 != 0:  # 每100个episode更新一次
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Progress - Episode {current_episode}', fontsize=16)
        
        episodes = training_history['episodes']
        
        # 1. 奖励曲线
        axes[0, 0].plot(episodes, training_history['rewards'], alpha=0.7)
        axes[0, 0].plot(episodes, self._smooth_curve(training_history['rewards']), 
                       color='red', linewidth=2, label='Smoothed')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 利润曲线
        axes[0, 1].plot(episodes, training_history['profits'], alpha=0.7)
        axes[0, 1].plot(episodes, self._smooth_curve(training_history['profits']), 
                       color='red', linewidth=2, label='Smoothed')
        axes[0, 1].set_title('Cumulative Profits')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Profit')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 平均持仓
        axes[0, 2].plot(episodes, training_history['positions'], alpha=0.7)
        axes[0, 2].plot(episodes, self._smooth_curve(training_history['positions']), 
                       color='red', linewidth=2, label='Smoothed')
        axes[0, 2].set_title('Average Position Size')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Position')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. 市场效率
        axes[1, 0].plot(episodes, training_history['market_efficiency'], alpha=0.7)
        axes[1, 0].plot(episodes, self._smooth_curve(training_history['market_efficiency']), 
                       color='red', linewidth=2, label='Smoothed')
        axes[1, 0].set_title('Market Efficiency')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Efficiency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. 价格波动率
        axes[1, 1].plot(episodes, training_history['volatility'], alpha=0.7)
        axes[1, 1].plot(episodes, self._smooth_curve(training_history['volatility']), 
                       color='red', linewidth=2, label='Smoothed')
        axes[1, 1].set_title('Price Volatility')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 6. 奖励分布
        if len(training_history['rewards']) > 50:
            recent_rewards = training_history['rewards'][-50:]
            axes[1, 2].hist(recent_rewards, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 2].axvline(np.mean(recent_rewards), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(recent_rewards):.2f}')
            axes[1, 2].set_title('Recent Rewards Distribution')
            axes[1, 2].set_xlabel('Reward')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'training_progress_{current_episode}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_episode_analysis(self, episode_data: Dict, episode: int):
        """绘制单个episode的详细分析"""
        if episode % 500 != 0:  # 每500个episode分析一次
            return
        
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(4, 3, figure=fig)
        
        # 数据准备
        steps = range(len(episode_data['prices']))
        prices = episode_data['prices']
        actions = episode_data['actions']
        positions = episode_data['positions']
        order_flows = episode_data['order_flows']
        rewards = np.cumsum(episode_data['rewards'])
        
        # 获取真实价值
        true_value = None
        if episode_data['market_states']:
            true_value = episode_data['market_states'][0].get('asset_true_value')
        
        # 1. 价格和真实价值
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(steps, prices, label='Market Price', linewidth=2)
        if true_value:
            ax1.axhline(y=true_value, color='red', linestyle='--', 
                       label=f'True Value: {true_value:.2f}')
        ax1.set_title(f'Price Evolution - Episode {episode}')
        ax1.set_xlabel('Auction Step')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 交易动作
        ax2 = fig.add_subplot(gs[1, 0])
        colors = ['red' if a > 0 else 'blue' for a in actions]
        ax2.bar(steps, actions, color=colors, alpha=0.7)
        ax2.set_title('Trading Actions')
        ax2.set_xlabel('Auction Step')
        ax2.set_ylabel('Order Size')
        ax2.grid(True)
        
        # 3. 持仓变化
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(steps, positions, linewidth=2, color='green')
        ax3.fill_between(steps, positions, alpha=0.3, color='green')
        ax3.set_title('Position Evolution')
        ax3.set_xlabel('Auction Step')
        ax3.set_ylabel('Position')
        ax3.grid(True)
        
        # 4. 累计奖励
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(steps, rewards, linewidth=2, color='purple')
        ax4.set_title('Cumulative Rewards')
        ax4.set_xlabel('Auction Step')
        ax4.set_ylabel('Cumulative Reward')
        ax4.grid(True)
        
        # 5. 订单流分析
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(steps, order_flows, linewidth=2, color='orange')
        ax5.set_title('Order Flow')
        ax5.set_xlabel('Auction Step')
        ax5.set_ylabel('Total Order Flow')
        ax5.grid(True)
        
        # 6. 价格-动作关系
        ax6 = fig.add_subplot(gs[2, 1])
        if true_value:
            price_signals = [p - true_value for p in prices]
            ax6.scatter(price_signals[:-1], actions, alpha=0.6, c=steps[:-1], cmap='viridis')
            ax6.set_xlabel('Price Signal (Price - True Value)')
            ax6.set_ylabel('Action')
            ax6.set_title('Action vs Price Signal')
            ax6.grid(True)
        
        # 7. 动作分布
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.hist(actions, bins=20, alpha=0.7, edgecolor='black')
        ax7.set_title('Action Distribution')
        ax7.set_xlabel('Action Size')
        ax7.set_ylabel('Frequency')
        ax7.grid(True)
        
        # 8. 市场影响分析
        ax8 = fig.add_subplot(gs[3, :])
        if len(prices) > 1:
            price_changes = np.diff(prices)
            ax8_twin = ax8.twinx()
            
            ax8.bar(steps[1:], price_changes, alpha=0.5, label='Price Changes', color='blue')
            ax8_twin.plot(steps[:-1], actions[:-1], color='red', linewidth=2, label='Actions')
            
            ax8.set_xlabel('Auction Step')
            ax8.set_ylabel('Price Change', color='blue')
            ax8_twin.set_ylabel('Action', color='red')
            ax8.set_title('Market Impact Analysis')
            ax8.grid(True)
            
            # 添加图例
            lines1, labels1 = ax8.get_legend_handles_labels()
            lines2, labels2 = ax8_twin.get_legend_handles_labels()
            ax8.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'episode_analysis_{episode}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_final_report(self, training_history: Dict):
        """创建最终训练报告"""
        # 创建综合报告图
        fig, axes = plt.subplots(3, 2, figsize=(16, 20))
        fig.suptitle('Final Training Report', fontsize=20)
        
        episodes = training_history['episodes']
        
        # 1. 学习曲线
        axes[0, 0].plot(episodes, training_history['rewards'], alpha=0.3, label='Raw')
        smoothed_rewards = self._smooth_curve(training_history['rewards'], window=50)
        axes[0, 0].plot(episodes, smoothed_rewards, linewidth=3, label='Smoothed')
        axes[0, 0].set_title('Learning Curve', fontsize=14)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 利润趋势
        axes[0, 1].plot(episodes, training_history['profits'], alpha=0.3, label='Raw')
        smoothed_profits = self._smooth_curve(training_history['profits'], window=50)
        axes[0, 1].plot(episodes, smoothed_profits, linewidth=3, label='Smoothed')
        axes[0, 1].set_title('Profit Trend', fontsize=14)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cumulative Profit')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 市场效率演化
        axes[1, 0].plot(episodes, training_history['market_efficiency'], alpha=0.3)
        smoothed_efficiency = self._smooth_curve(training_history['market_efficiency'], window=50)
        axes[1, 0].plot(episodes, smoothed_efficiency, linewidth=3)
        axes[1, 0].set_title('Market Efficiency Evolution', fontsize=14)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Market Efficiency')
        axes[1, 0].grid(True)
        
        # 4. 波动率分析
        axes[1, 1].plot(episodes, training_history['volatility'], alpha=0.3)
        smoothed_volatility = self._smooth_curve(training_history['volatility'], window=50)
        axes[1, 1].plot(episodes, smoothed_volatility, linewidth=3)
        axes[1, 1].set_title('Price Volatility', fontsize=14)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].grid(True)
        
        # 5. 性能统计
        axes[2, 0].remove()
        ax_stats = fig.add_subplot(3, 2, 5)
        
        # 计算统计数据
        final_rewards = training_history['rewards'][-100:] if len(training_history['rewards']) >= 100 else training_history['rewards']
        stats_text = f"""
        Training Statistics (Last 100 Episodes):
        
        Mean Reward: {np.mean(final_rewards):.2f}
        Std Reward: {np.std(final_rewards):.2f}
        Max Reward: {np.max(final_rewards):.2f}
        Min Reward: {np.min(final_rewards):.2f}
        
        Final Market Efficiency: {training_history['market_efficiency'][-1]:.3f}
        Final Volatility: {training_history['volatility'][-1]:.3f}
        
        Total Episodes: {len(episodes)}
        """
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, 
                     fontsize=12, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis('off')
        
        # 6. 最终奖励分布
        axes[2, 1].hist(final_rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[2, 1].axvline(np.mean(final_rewards), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(final_rewards):.2f}')
        axes[2, 1].set_title('Final Reward Distribution', fontsize=14)
        axes[2, 1].set_xlabel('Reward')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'final_training_report.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建交互式图表
        self._create_interactive_plots(training_history)
    
    def _create_interactive_plots(self, training_history: Dict):
        """创建交互式图表"""
        # 创建子图
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Learning Curve', 'Market Efficiency', 'Volatility', 'Position Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        episodes = training_history['episodes']
        
        # 学习曲线
        fig.add_trace(
            go.Scatter(x=episodes, y=training_history['rewards'], 
                      mode='lines', name='Rewards', opacity=0.3),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=episodes, y=self._smooth_curve(training_history['rewards']), 
                      mode='lines', name='Smoothed Rewards', line=dict(width=3)),
            row=1, col=1
        )
        
        # 市场效率
        fig.add_trace(
            go.Scatter(x=episodes, y=training_history['market_efficiency'], 
                      mode='lines', name='Market Efficiency'),
            row=1, col=2
        )
        
        # 波动率
        fig.add_trace(
            go.Scatter(x=episodes, y=training_history['volatility'], 
                      mode='lines', name='Volatility'),
            row=2, col=1
        )
        
        # 持仓分析
        fig.add_trace(
            go.Scatter(x=episodes, y=training_history['positions'], 
                      mode='lines', name='Avg Position'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Interactive Training Analysis",
            showlegend=True,
            height=800
        )
        
        # 保存交互式图表
        plot(fig, filename=os.path.join(self.plots_dir, 'interactive_analysis.html'), auto_open=False)
    
    def _smooth_curve(self, data: List, window: int = 20) -> List:
        """平滑曲线"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start_idx:end_idx]))
        
        return smoothed
    
    def plot_strategy_comparison(self, results: Dict):
        """比较不同策略的性能"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Strategy Comparison', fontsize=16)
        
        strategies = list(results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
        
        # 1. 累计奖励比较
        for i, (strategy, data) in enumerate(results.items()):
            axes[0, 0].plot(data['episodes'], np.cumsum(data['rewards']), 
                           label=strategy, color=colors[i], linewidth=2)
        axes[0, 0].set_title('Cumulative Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 平均奖励比较
        avg_rewards = [np.mean(data['rewards'][-100:]) for data in results.values()]
        std_rewards = [np.std(data['rewards'][-100:]) for data in results.values()]
        
        x_pos = np.arange(len(strategies))
        axes[0, 1].bar(x_pos, avg_rewards, yerr=std_rewards, capsize=5, 
                      color=colors, alpha=0.7)
        axes[0, 1].set_title('Average Reward (Last 100 Episodes)')
        axes[0, 1].set_xlabel('Strategy')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(strategies, rotation=45)
        axes[0, 1].grid(True)
        
        # 3. 市场效率比较
        for i, (strategy, data) in enumerate(results.items()):
            axes[1, 0].plot(data['episodes'], data['market_efficiency'], 
                           label=strategy, color=colors[i], linewidth=2)
        axes[1, 0].set_title('Market Efficiency')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Market Efficiency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 风险-收益分析
        for i, (strategy, data) in enumerate(results.items()):
            rewards = data['rewards'][-100:]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            axes[1, 1].scatter(std_reward, mean_reward, s=100, 
                             color=colors[i], label=strategy, alpha=0.7)
        
        axes[1, 1].set_title('Risk-Return Analysis')
        axes[1, 1].set_xlabel('Risk (Std of Rewards)')
        axes[1, 1].set_ylabel('Return (Mean Reward)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'strategy_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_report(results_dir: str):
    """创建总结报告"""
    # 这里可以添加更多的分析和报告生成功能
    print(f"Summary report created in {results_dir}")