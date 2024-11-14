import torch
from torchrl import Agent, Environment

# Create an environment
env = Environment("CartPole-v1")

# Initialize an agent
agent = Agent(env)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
