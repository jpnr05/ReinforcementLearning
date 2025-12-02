import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Importa a tua função factory do teu código
# Certifica-te que as classes CustomAssaultObservation e CustomAssaultReward estão acessíveis
from AmbienteAssault import make_custom_assault 

# 1. Configuração
# O PPO gosta de ambientes vetorizados (vários a correr ao mesmo tempo)
# Aqui usamos DummyVecEnv para envolver o teu ambiente customizado
env = DummyVecEnv([lambda: make_custom_assault(render_mode="rgb_array", enable_noise=True)])

# 2. Criar o Modelo
# Usamos "CnnPolicy" porque o input são pixels (imagem)
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99
)

print("--- A iniciar treino com PPO ---")
# 3. Treinar (timesteps define a duração do treino)
# Para ver resultados reais em Atari, precisas de 1M a 10M de steps. 
# 10.000 é só para testar se o código corre.
model.learn(total_timesteps=10000)

# 4. Guardar
model.save("ppo_assault_custom")
print("Modelo PPO guardado.")

# 5. Testar / Visualizar
# Para visualizar, recriamos o ambiente com render_mode='human'
env_visual = make_custom_assault(render_mode="human", enable_noise=False)
obs, _ = env_visual.reset()

print("A visualizar o agente treinado...")
for _ in range(1000):
    # O modelo prevê a ação baseada na observação
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_visual.step(action)
    if terminated or truncated:
        obs, _ = env_visual.reset()

env_visual.close()