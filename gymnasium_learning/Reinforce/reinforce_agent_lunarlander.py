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
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
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
        
        m = Categorical(action_probs)
        action = m.sample()
        
        self.log_probs_history.append(m.log_prob(action))
        return action.item()

    def update_policy(self):
        policy_loss = []
        
        returns = []
        G = 0
        for r in reversed(self.rewards_history):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)
        
        # 标准化回报 (有助于稳定训练，特别是对于回报范围较大的环境)
        if len(returns) > 1: # 避免除以零
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for log_prob, Gt in zip(self.log_probs_history, returns):
            policy_loss.append(-log_prob * Gt)
        
        self.optimizer.zero_grad()
        torch.stack(policy_loss).sum().backward()
        self.optimizer.step()

        self.rewards_history = []
        self.log_probs_history = []

# 3. 训练 REINFORCE Agent
def train():
    env = gym.make("LunarLander-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim)

    num_episodes = 2000 # 增加训练回合数
    print_interval = 10 # 每 10 个回合打印一次信息

    scores = deque(maxlen=100) # 记录最近 100 个回合的分数

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

        agent.update_policy()
        scores.append(total_reward)

        avg_score = np.mean(scores)

        if (episode + 1) % print_interval == 0:
            print(f"Episode {episode + 1}, Avg Score (100 episodes): {avg_score:.2f}")
        
        # LunarLander-v2 的目标是达到 200 分
        if avg_score >= 200:
            print(f"环境解决于 {episode + 1} 回合! 平均分数: {avg_score:.2f}")
            break

    env.close()

if __name__ == "__main__":
    train()