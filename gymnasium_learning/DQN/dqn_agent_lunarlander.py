import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# 1. 定义 Q 网络 (Q-Network)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256) # 增加网络宽度
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. 定义经验回放记忆库 (Replay Buffer)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 确保缓冲区中的样本数量至少等于批次大小
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# 3. 定义 DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer_capacity=100000, batch_size=64, gamma=0.99, lr=0.0005, target_update_freq=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval() # 目标网络不进行训练

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.update_counter = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                return q_values.argmax().item()

    def update_model(self):
        transitions = self.replay_buffer.sample(self.batch_size)
        if transitions is None: # 如果样本不足，不更新
            return

        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device) # True表示回合结束，False表示未结束

        # 计算当前 Q 值：Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions)

        # 计算目标 Q 值：r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            # next_q_values = self.target_q_network(next_states).max(1)[0].unsqueeze(1) # [0] 获取最大值，[1] 获取索引
            # 使用 target_q_network 预测下一个状态的最大 Q 值
            next_q_values = self.target_q_network(next_states).max(1, keepdim=True)[0] # keepdim=True 保持维度

            # 如果 done 为 True，则下一状态没有未来奖励，target_q_values = reward
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 (防止梯度爆炸)
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

# 4. 训练 DQN Agent
def train():
    env = gym.make("LunarLander-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 2000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995 # 适当降低衰减率，给更多探索时间
    epsilon = epsilon_start

    scores = deque(maxlen=100) # 记录最近 100 个回合的分数

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update_model()

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        scores.append(total_reward)

        avg_score = np.mean(scores)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Avg Score (100 episodes): {avg_score:.2f}, Epsilon: {epsilon:.2f}")
        
        # LunarLander-v2 的目标是达到 200 分
        if avg_score >= 200:
            print(f"环境解决于 {episode + 1} 回合! 平均分数: {avg_score:.2f}")
            # 可以选择保存模型
            # torch.save(agent.q_network.state_dict(), 'dqn_lunarlander_solved.pth')
            break

    env.close()

if __name__ == "__main__":
    train()