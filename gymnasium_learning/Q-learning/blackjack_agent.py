import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        action_space_n: int,
        discount_factor: float = 0.95,
    ):
        """
        初始化 Q-learning 智能体.

        参数:
        learning_rate (float): 学习率.
        initial_epsilon (float): 初始探索率.
        epsilon_decay (float): Epsilon 的衰减率.
        final_epsilon (float): 最终探索率.
        action_space_n (int): 动作空间的数量.
        discount_factor (float): 折扣因子.
        """
        self.q_values = defaultdict(lambda: np.zeros(action_space_n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool], action_space) -> int:
        """
        使用 epsilon-greedy 策略选择一个动作.
        """
        if np.random.random() < self.epsilon:
            return action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """
        更新 Q-value.
        """
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

if __name__ == "__main__":
    # 环境设置
    env = gym.make("Blackjack-v1", sab=True)

    # 训练参数
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    agent = BlackjackAgent(
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        action_space_n=env.action_space.n,
    )

    # 训练循环
    rewards = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(obs, env.action_space)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
        agent.decay_epsilon()

    # 绘制结果
    rolling_window = 1000
    plt.figure(figsize=(12, 5))
    
    # 计算移动平均值
    moving_average = np.convolve(np.array(rewards), np.ones(rolling_window)/rolling_window, mode='valid')
    # 为移动平均值创建对齐的 x 轴
    moving_average_x = range(rolling_window - 1, len(rewards))

    plt.plot(range(len(rewards)), rewards, label="Raw Reward", alpha=0.3)
    plt.plot(moving_average_x, moving_average, label=f"Moving Average (window={rolling_window})", color='red')
    plt.title("Blackjack Q-Learning Agent Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 演示训练好的智能体
    print("\n--- 演示训练好的智能体 ---")
    for _ in range(5):
        obs, info = env.reset()
        done = False
        print(f"开始新一局. 初始状态: {obs}")
        while not done:
            # 在演示时，我们通常使用贪心策略（epsilon=0）
            agent.epsilon = 0
            action = agent.get_action(obs, env.action_space)
            print(f"状态: {obs}, 动作: {'停牌' if action == 0 else '要牌'}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(f"游戏结束. 最终状态: {obs}, 奖励: {reward}\n")

    env.close()