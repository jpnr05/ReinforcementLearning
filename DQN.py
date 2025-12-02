import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Importa a tua função
from AmbienteAssault import make_custom_assault 

# 1. Configuração
env = DummyVecEnv([lambda: make_custom_assault(render_mode="rgb_array", enable_noise=True)])

# 2. Criar o Modelo DQN
# buffer_size: quantos passos do passado ele guarda na memória (Cuidado com a RAM)
# learning_starts: quantos passos aleatórios antes de começar a aprender
model = DQN(
    "CnnPolicy", 
    env, 
    verbose=1,
    buffer_size=10000,  # Em treino sério, usa 100.000 ou 1.000.000
    learning_rate=1e-4,
    learning_starts=1000,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.1, # 10% do tempo a explorar no início
    exploration_final_eps=0.01 # No fim, explora 1% das vezes
)

print("--- A iniciar treino com DQN ---")
model.learn(total_timesteps=10000)

model.save("dqn_assault_custom")
print("Modelo DQN guardado.")