import gymnasium as gym
import ale_py
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Importa a tua factory do ambiente customizado
from AmbienteAssault import make_custom_assault

def run_evaluation():
    # --- CONFIGURAÇÃO ---
    nome_modelo = "ppo_assault_custom_parallel" # O nome do ficheiro .zip (sem a extensão)
    caminho_arquivo = f"{nome_modelo}.zip"
    n_episodios_metricas = 20  # Quantos jogos usar para calcular a média (Matemática)
    n_episodios_visuais = 3    # Quantos jogos queres ver com os teus olhos

    # 1. Verificar se o modelo existe
    if not os.path.exists(caminho_arquivo):
        print(f" ERRO: O ficheiro '{caminho_arquivo}' não foi encontrado.")
        print("Certifica-te que já treinaste o modelo e o ficheiro está na mesma pasta.")
        return

    print(f" Modelo '{nome_modelo}' encontrado. A carregar...")

    # 2. Carregar o Modelo
    # Precisamos de um ambiente dummy só para carregar a arquitetura do modelo
    # enable_noise=False para testar a performance "limpa" (ou True se quiseres testar robustez)
    env_dummy = make_custom_assault(render_mode="rgb_array", enable_noise=False)
    
    try:
        model = PPO.load(caminho_arquivo, env=env_dummy)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    print("\n" + "="*40)
    print("   FASE 1: CÁLCULO DE MÉTRICAS (Rápido)")
    print("="*40)
    
    # Para métricas precisas, usamos o evaluate_policy do Stable-Baselines3
    # Ele corre o jogo n vezes e retorna a média e o desvio padrão
    # deterministic=True faz o agente usar a melhor ação possível (sem exploração aleatória)
    mean_reward, std_reward = evaluate_policy(
        model, 
        env_dummy, 
        n_eval_episodes=n_episodios_metricas, 
        deterministic=True,
        return_episode_rewards=False
    )
    
    print(f" RESULTADOS (Baseados em {n_episodios_metricas} jogos):")
    print(f"   🔹 Média de Recompensa: {mean_reward:.2f}")
    print(f"   🔹 Estabilidade (Desvio Padrão): +/- {std_reward:.2f}")
    
    if mean_reward > 0:
        print("   O agente parece estar a jogar bem!")
    else:
        print("   O agente está com pontuação baixa. Talvez precise de mais treino.")

    env_dummy.close()

    # ---------------------------------------------------------
    
    print("\n" + "="*40)
    print("   FASE 2: MODO ESPECTADOR (Visual)")
    print("="*40)
    print("A abrir janela do jogo... (Pode demorar 2 segundos)")

    # Cria um novo ambiente COM render_mode='human' para aparecer a janela
    env_visual = make_custom_assault(render_mode="human", enable_noise=False)
    
    scores_visuais = []

    try:
        for i in range(n_episodios_visuais):
            obs, _ = env_visual.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                # O modelo prevê a ação
                action, _ = model.predict(obs, deterministic=True)
                
                # O ambiente executa
                obs, reward, terminated, truncated, info = env_visual.step(action)
                
                # Como o teu reward wrapper divide por 100, a pontuação real do jogo 
                # pode estar escondida. O 'reward' aqui é o valor modificado.
                total_reward += reward
                steps += 1
                
                done = terminated or truncated
            
            scores_visuais.append(total_reward)
            print(f"Jogo {i+1}/{n_episodios_visuais} terminado. | Steps: {steps} | Reward Modificado: {total_reward:.4f}")

    except KeyboardInterrupt:
        print("\nInterrompido pelo utilizador.")
    finally:
        env_visual.close()
        print("\nTeste concluído.")

if __name__ == "__main__":
    run_evaluation()