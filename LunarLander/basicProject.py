import gymnasium as gym
from stable_baselines3 import PPO
import os

env = gym.make('LunarLander-v2')
env.reset()

models_dir = "models/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model = PPO('MlpPolicy', env, verbose=1)

iters = 0
TimeSteps = 10_000

while True:
    iters += 1
    model.learn(total_timesteps=TimeSteps,reset_num_timesteps=False)
    model.save(f"{models_dir}/{TimeSteps*iters}")