import gymnasium as gym
import ale_py  # <--- Isto é CRÍTICO na nova versão
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from AmbienteAssault import CustomAssaultObservation, CustomAssaultReward

# --- 3. Função Construtora (Factory) ---
def make_custom_assault(render_mode="rgb_array", enable_noise=True):
    """
    Cria o ambiente Assault com as personalizações aplicadas.
    Args:
        render_mode: 'human' para veres jogar, 'rgb_array' para treino rápido.
        enable_noise: True para adicionar grão na imagem, False para imagem limpa.
    """
    # Carrega o ambiente original
    env = gym.make("ALE/Assault-v5", render_mode=render_mode)
    
    # Aplica os Wrappers
    env = CustomAssaultObservation(env, enable_noise=enable_noise)
    env = CustomAssaultReward(env)
    
    return env

# --- 4. Bloco de Teste (Para verificar se funciona) ---
if __name__ == "__main__":
    # Teste com Ruído LIGADO
    print("A criar ambiente com ruído...")
    env = make_custom_assault(render_mode="human", enable_noise=True)
    
    obs, info = env.reset() 
    
    # Simula 200 passos aleatórios para testar se não crasha
    for _ in range(200):
        # Ação aleatória (0 a 6 no Assault)
        action = env.action_space.sample()
        
        # Executa o passo
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Mostra o formato da observação (deve ser 210x160, sem o 3 do RGB)
        if _ == 0:
            print(f"Formato da observação recebida: {obs.shape}")
            print(f"Exemplo de reward modificado: {reward}")

        if terminated or truncated:
            obs, info = env.reset()
            
    env.close()
    print("Teste concluído com sucesso!")