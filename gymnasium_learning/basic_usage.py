import gymnasium as gym

# 创建一个环境
env = gym.make("CartPole-v1", render_mode="human")

# 重置环境，获取初始观测
observation, info = env.reset(seed=42)

for _ in range(1000):
    # 从动作空间中随机选择一个动作
    action = env.action_space.sample()
    
    # 执行动作，获取下一个状态、奖励、是否结束等信息
    observation, reward, terminated, truncated, info = env.step(action)

    # 如果环境结束，则重置
    if terminated or truncated:
        observation, info = env.reset()

# 关闭环境
env.close()