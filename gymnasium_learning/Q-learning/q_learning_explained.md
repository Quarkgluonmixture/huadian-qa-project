# Q-learning 原理与21点游戏实现

## 1. Q-learning 核心概念
```mermaid
graph LR
A[状态 s] --> B[选择动作 a]
B --> C[获得奖励 r]
C --> D[转移到新状态 s']
D --> E[更新 Q(s,a)]
```

### 关键公式
Q值更新公式：
`Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]`

其中：
- α (alpha)：学习率 (0.01)
- γ (gamma)：折扣因子 (0.95)
- max_a' Q(s',a')：新状态下最大预期收益

## 2. 21点游戏中的Q-learning实现

### 状态表示 (State)
在 [`BlackjackAgent`](gymnasium_learning/blackjack_agent.py:33) 中：
```python
# 状态: (玩家点数, 庄家明牌点数, 是否有A)
obs: tuple[int, int, bool]
```

### 动作空间 (Action)
在 [`env.action_space`](gymnasium_learning/blackjack_agent.py:38) 中定义：
- 0 = 停牌 (STAND)
- 1 = 要牌 (HIT)

### Q值存储
使用 [`defaultdict`](gymnasium_learning/blackjack_agent.py:25) 初始化Q表：
```python
self.q_values = defaultdict(lambda: np.zeros(action_space_n))
```

### 策略选择
ε-greedy 策略 [`get_action`](gymnasium_learning/blackjack_agent.py:33)：
```python
def get_action(self, obs, action_space):
    if np.random.random() < self.epsilon:  # 探索
        return action_space.sample()
    else:  # 利用
        return int(np.argmax(self.q_values[obs]))
```

### Q值更新
核心更新逻辑 [`update`](gymnasium_learning/blackjack_agent.py:42)：
```python
future_q = (not terminated) * np.max(self.q_values[next_obs])
td_error = reward + self.discount_factor * future_q - self.q_values[obs][action]
self.q_values[obs][action] += self.lr * td_error
```

## 3. 训练过程分析

### 参数设置
```python
learning_rate = 0.01
n_episodes = 100_000
epsilon_decay = start_epsilon / (n_episodes / 2)  # 线性衰减
```

### 训练循环
```python
for episode in range(n_episodes):
    # 重置环境
    while not done:
        # 选择动作
        # 执行动作
        # 更新Q值
    # 衰减ε
    agent.decay_epsilon()
```

## 4. 结果解读

### 奖励曲线
![训练曲线](blackjack_agent.py中的绘图代码)

- **原始奖励**：剧烈波动，反映探索过程
- **移动平均**：展示学习趋势
- 理想收敛值 ≈ 0（对应~50%胜率）

### 策略演示
```python
# 演示时设置ε=0 (纯贪婪策略)
agent.epsilon = 0
action = agent.get_action(obs, env.action_space)
```

## 5. 运行指南

1. 安装依赖：
```bash
pip install gymnasium numpy matplotlib
```

2. 运行程序：
```bash
python gymnasium_learning/blackjack_agent.py
```

3. 观察输出：
- 训练过程中的实时奖励曲线
- 训练后的5局游戏演示

## 6. Q-learning 学习要点

1. **探索 vs 利用**：ε控制探索新策略的概率
2. **时间差分**：基于当前估计更新Q值
3. **奖励设计**：稀疏奖励(+1/-1)下的学习挑战
4. **状态表示**：离散状态空间的有效性