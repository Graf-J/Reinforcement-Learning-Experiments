import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset(seed=123, options={"low": -0.1, "high": 0.1})
env.render()
