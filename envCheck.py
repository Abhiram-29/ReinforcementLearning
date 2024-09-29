from stable_baselines3.common.env_checker import check_env
from snakeEnv import SnakeEnv

env = SnakeEnv()
print("Checking the environment")
check_env(env)