import gymnasium as gym
import ale_py 
import numpy as np
from gymnasium import spaces

# --- 1. Wrapper para Alterar o Reward (Recompensa) ---
class CustomAssaultReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        # 1. Normalização: Reduzimos o valor para manter a estabilidade numérica (escala 0.01)
        modified_reward = reward * 0.01 
        
        # 2. Survival Bonus: Se o agente estiver vivo (mesmo sem matar), recebe um pequeno incentivo.
        if reward == 0:
            modified_reward += 0.001
            
        return modified_reward

# --- 2. Wrapper para Alterar a Observação (Visão) ---
class CustomAssaultObservation(gym.ObservationWrapper):
    def __init__(self, env, enable_noise=True):
        super().__init__(env)
        self.enable_noise = enable_noise
        
        # --- CORREÇÃO CRÍTICA AQUI ---
        # O Stable-Baselines3 exige 3 dimensões: (Altura, Largura, Canais)
        # Mesmo sendo preto e branco, temos de dizer que há 1 canal.
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(210, 160, 1), # <--- Adicionado o ", 1"
            dtype=np.uint8
        )

    def observation(self, observation):
        # Passo A: Converter para Preto e Branco (Média dos canais de cor)
        # axis=2 faz a média dos 3 canais de cor (R, G, B) resultando em (210, 160)
        grayscale = np.mean(observation, axis=2).astype(np.uint8)
        
        # Passo B: Adicionar Ruído (Opcional)
        if self.enable_noise:
            noise = np.random.randint(0, 30, grayscale.shape, dtype=np.uint8)
            final_obs_2d = np.clip(grayscale + noise, 0, 255).astype(np.uint8)
        else:
            final_obs_2d = grayscale
            
        # --- CORREÇÃO CRÍTICA AQUI ---
        # Transformar (210, 160) em (210, 160, 1)
        # Isto cria a dimensão "falsa" de cor para satisfazer a CnnPolicy
        final_obs_3d = np.expand_dims(final_obs_2d, axis=-1)
        
        return final_obs_3d

# --- 3. Função Construtora (Factory) ---
def make_custom_assault(render_mode="rgb_array", enable_noise=True):
    """
    Cria o ambiente Assault com as personalizações aplicadas.
    """
    # Importante: Registar o ale_py dentro da factory para garantir compatibilidade com multiprocessos
    gym.register_envs(ale_py)
    
    # Carrega o ambiente original
    env = gym.make("ALE/Assault-v5", render_mode=render_mode)
    
    # Aplica os Wrappers
    env = CustomAssaultObservation(env, enable_noise=enable_noise)
    env = CustomAssaultReward(env)
    
    return env

# --- 4. Bloco de Teste ---
if __name__ == "__main__":
    print("A criar ambiente corrigido...")
    # Teste rápido sem renderização visual pesada
    env = make_custom_assault(render_mode="rgb_array", enable_noise=True)
    
    obs, info = env.reset() 
    
    print(f"Formato da observação (Deve ser 210, 160, 1): {obs.shape}")
    
    if obs.shape == (210, 160, 1):
        print("CORRETO! O formato é compatível com CnnPolicy.")
    else:
        print("ERRO! O formato ainda está errado.")

    env.close()