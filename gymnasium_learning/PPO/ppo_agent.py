import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque

# 1. 定义 Actor 和 Critic 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 输出动作的对数概率（log_softmax）
        return torch.log_softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1) # 输出状态价值 V(s)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, clip_epsilon=0.2, ppo_epochs=10, batch_size=64, entropy_coef=0.01):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = deque()

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.memory.append((state, action, log_prob, value, reward, done))

    def clear_memory(self):
        self.memory.clear()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            log_probs = self.actor(state)
            value = self.critic(state)
        
        # 从对数概率中创建分类分布
        m = Categorical(log_probs.exp()) # exp() 得到概率
        action = m.sample()
        
        return action.item(), m.log_prob(action).item(), value.item()

    def compute_advantages_and_returns(self):
        states, actions, old_log_probs, values, rewards, dones = zip(*self.memory)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算回报 (Returns)
        returns = torch.zeros_like(rewards).to(self.device)
        R = 0
        for i in reversed(range(len(rewards))):
            if dones[i]: # 如果回合终止，R归零
                R = 0
            R = rewards[i] + self.gamma * R
            returns[i] = R
        
        # 计算优势函数 (Advantages)
        # 这里使用简单的 TD 优势 (GAE 更复杂，但效果更好)
        advantages = returns - values
        
        # 标准化优势函数 (重要)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, old_log_probs, values, advantages, returns

    def update_policy(self):
        states, actions, old_log_probs, values, advantages, returns = self.compute_advantages_and_returns()
        
        # PPO 训练 K 个 epochs
        for _ in range(self.ppo_epochs):
            # 随机打乱数据并分成 mini-batch
            indices = torch.randperm(len(states))
            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Actor 网络：计算当前策略的对数概率
                current_log_probs = self.actor(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # Critic 网络：计算当前状态价值
                current_values = self.critic(batch_states).squeeze(1)

                # 1. 计算重要性采样比率 (ratio)
                ratio = torch.exp(current_log_probs - batch_old_log_probs)

                # 2. 计算策略损失 (Clipped Policy Loss)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 3. 计算价值损失 (Value Loss)
                critic_loss = nn.MSELoss()(current_values, batch_returns).mean()

                # 4. (可选) 熵损失 (Entropy Loss)
                # 这有助于鼓励探索，防止策略过于确定
                m = Categorical(self.actor(batch_states).exp())
                entropy_loss = m.entropy().mean()
                
                # 总损失
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy_loss

                # 更新网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        self.clear_memory() # 清空记忆库，因为 PPO 是 On-policy 算法

# 3. 训练 PPO Agent
def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)

    num_episodes = 2000 # 增加训练回合数
    steps_per_batch = 2048 # 每收集 2048 步数据就更新一次策略
    print_interval = 10 # 每 10 次策略更新打印一次信息

    episode_rewards = deque(maxlen=print_interval) # 记录最近的奖励

    current_steps = 0
    num_updates = 0

    state, _ = env.reset()

    for episode in range(1, num_episodes + 1):
        done = False
        episode_reward = 0

        while not done and current_steps < steps_per_batch:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, log_prob, value, reward, done)
            
            state = next_state
            episode_reward += reward
            current_steps += 1

            if done:
                state, _ = env.reset() # 如果回合结束，重置环境
        
        episode_rewards.append(episode_reward)

        if current_steps >= steps_per_batch or done: # 收集到足够数据或者回合结束
            agent.update_policy()
            current_steps = 0 # 重置步数
            num_updates += 1
            
            if num_updates % print_interval == 0:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                print(f"Update {num_updates}, Avg Episode Reward: {avg_reward:.2f}")
                episode_rewards.clear() # 清空奖励记录

    env.close()

if __name__ == "__main__":
    train()