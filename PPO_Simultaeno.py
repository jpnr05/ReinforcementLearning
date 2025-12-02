import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from AmbienteAssault import make_custom_assault 
from stable_baselines3.common.vec_env import VecFrameStack
# --- BLOCO PRINCIPAL (OBRIGATÓRIO NO WINDOWS) ---
if __name__ == "__main__":
    
    # 1. Configuração Paralela
    # n_envs=n: Vai abrir 4 jogos ao mesmo tempo (ajusta conforme os teus núcleos de CPU)
    # vec_env_cls=SubprocVecEnv: Garante que cada jogo corre num processo separado (True Parallelism)
    print("A preparar 8 ambientes paralelos...")
    env = make_vec_env(
        lambda: make_custom_assault(render_mode="rgb_array", enable_noise=True),
        n_envs=8, 
        vec_env_cls=SubprocVecEnv
    )

    # 2. APLICAR FRAME STACKING AQUI
    # Isto empilha 4 frames consecutivos. 
    # O input da rede passa de (210, 160, 1) para (210, 160, 4)
    env = VecFrameStack(env, n_stack=4)

    # 2. Criar o Modelo
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )

    print("--- A iniciar treino com PPO (Paralelo) ---")
    
    # 3. Treinar com Barra de Progresso
    #progress_bar=True (Requer a biblioteca tqdm instalada: pip install tqdm rich) só estético 
    model.learn(total_timesteps=100000, progress_bar=True)

    # 4. Guardar
    model.save("ppo_assault_custom_parallel_comStack")
    print("Modelo PPO guardado.")
    
    # Fechar os ambientes de treino para libertar memória
    env.close()

    # -------------------------------------------------
    # 5. Testar / Visualizar (Isto corre APENAS num ambiente, para se ver o resultado)
    # -------------------------------------------------
    print("A carregar ambiente visual para teste...")
    env_visual = make_custom_assault(render_mode="human", enable_noise=False)
    
    # Carregar o modelo que acabaste de treinar
    # (Opcional se o 'model' ainda estiver em memória, mas boa prática)
    model = PPO.load("ppo_assault_custom_parallel")

    obs, _ = env_visual.reset()

    print("A visualizar o agente a jogar...")
    try:
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_visual.step(action)
            
            if terminated or truncated:
                obs, _ = env_visual.reset()
    except KeyboardInterrupt:
        print("Teste interrompido pelo utilizador.")
    finally:
        env_visual.close()
        print("Fim.")