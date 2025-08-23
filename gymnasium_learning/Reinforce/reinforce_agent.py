import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# 1. 定义策略网络 (Policy Network)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 对于离散动作空间，使用 softmax 得到动作概率
        return torch.softmax(self.fc3(x), dim=-1)

# 2. 定义 REINFORCE Agent
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_network = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
        self.rewards_history = []
        self.log_probs_history = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state)
        
        # 创建一个分类分布，从中采样动作
        m = Categorical(action_probs)
        action = m.sample()
        
        # 存储对数概率和动作
        self.log_probs_history.append(m.log_prob(action))
        return action.item()

    def update_policy(self):
        policy_loss = []
        
        # 计算折扣累积回报 (G_t)
        returns = []
        G = 0
        for r in reversed(self.rewards_history):
            G = r + self.gamma * G
            returns.insert(0, G) # 插入到列表开头，保持时间顺序

        returns = torch.FloatTensor(returns).to(self.device)
        
        # 标准化回报 (可选，但通常有助于稳定训练)
        # if len(returns) > 1: # 避免除以零
        #     returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for log_prob, Gt in zip(self.log_probs_history, returns):
            # 策略梯度损失：- log_prob * G_t
            policy_loss.append(-log_prob * Gt)
        
        self.optimizer.zero_grad()
        # 将所有损失求和并反向传播
        torch.stack(policy_loss).sum().backward()
        self.optimizer.step()

        # 清空历史记录
        self.rewards_history = []
        self.log_probs_history = []

# 3. 训练 REINFORCE Agent
def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim)

    num_episodes = 1000 # 增加训练回合数
    print_interval = 50 # 每 50 个回合打印一次信息

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.rewards_history.append(reward)
            state = next_state
            total_reward += reward

        # 每个回合结束后更新策略
        agent.update_policy()

        if (episode + 1) % print_interval == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    train()