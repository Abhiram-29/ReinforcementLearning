import gymnasium as gym
from stable_baselines3 import PPO
from icecream import ic
env = gym.make("LunarLander-v2", render_mode="human")

models_dir = "models/PPO"
model_path = f"{models_dir}/270000.zip"
model = PPO.load(model_path,env=env)
venv = model.get_env()
obs = venv.reset()

ic(venv.reset())

done = False

while not done:
    action, _ = model.predict(obs)
    obs, rewards, done, info = venv.step(action)
    ic(obs,done)
    venv.render()
venv.close()
env.close()