import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# 1. 定义 Actor 网络 (Policy Network)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 输出动作的概率分布
        return torch.softmax(self.fc3(x), dim=-1)

# 2. 定义 Critic 网络 (Value Network)
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # 输出状态价值 V(s)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 3. 定义 A2C Agent
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99, entropy_coef=0.01):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.log_probs = []
        self.rewards = []
        self.values = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Actor 预测动作概率
        action_probs = self.actor(state)
        m = Categorical(action_probs)
        action = m.sample()
        
        # Critic 预测状态价值
        value = self.critic(state)

        self.log_probs.append(m.log_prob(action))
        self.values.append(value)
        
        return action.item()

    def update_networks(self, next_state, done):
        # 计算下一个状态的价值
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        next_value = self.critic(next_state)
        
        # 计算每个时间步的 G_t (蒙特卡洛回报) 或 TD 目标
        # 这里使用 TD(0) 目标来计算优势函数
        returns = []
        R = next_value.item() * (1 - done) # 如果是终止状态，R 为 0
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 清空梯度
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # 将列表转换为 Tensor
        log_probs = torch.stack(self.log_probs).to(self.device)
        values = torch.stack(self.values).to(self.device)

        # 计算优势函数 (Advantage)
        advantages = returns - values.squeeze()

        # 计算 Actor 损失 (Policy Loss)
        actor_loss = -(log_probs * advantages.detach()).sum() # detach 阻止梯度流向 Critic
        
        # 添加熵正则化项
        entropy_loss = - (log_probs.exp() * log_probs).sum() # 策略熵
        actor_loss = actor_loss - self.entropy_coef * entropy_loss

        # 计算 Critic 损失 (Value Loss)
        critic_loss = nn.MSELoss()(values.squeeze(), returns).sum() # 最小化价值网络的预测与实际回报的误差

        # 反向传播并更新
        (actor_loss + critic_loss).backward() # 两个损失一起反向传播
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # 清空历史记录
        self.log_probs = []
        self.rewards = []
        self.values = []

# 4. 训练 A2C Agent
def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim)

    num_episodes = 2000 # 增加训练回合数
    print_interval = 100 # 每 100 个回合打印一次信息

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.rewards.append(reward)
            state = next_state
            total_reward += reward

        # 每个回合结束后更新网络
        agent.update_networks(state, done) # 传入最后的状态和 done 标志

        if (episode + 1) % print_interval == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    train()