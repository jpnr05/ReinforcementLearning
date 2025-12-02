import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from AmbienteAssault import make_custom_assault 
import os  

if __name__ == "__main__":
    
    # Nome do ficheiro onde guardas o modelo
    nome_modelo = "ppo_assault_custom_parallel"
    caminho_arquivo = f"{nome_modelo}.zip"

    # 1. Configurar Ambientes
    print("A preparar 8 ambientes paralelos...")
    env = make_vec_env(
        lambda: make_custom_assault(render_mode="rgb_array", enable_noise=True),
        n_envs=8, 
        vec_env_cls=SubprocVecEnv
    )

    # 2. Lógica Inteligente: Carregar ou Criar
    if os.path.exists(caminho_arquivo):
        print(f"--- Encontrei um treino anterior: {nome_modelo} ---")
        print("A carregar modelo e a continuar o treino...")
        
        # Carrega o modelo e liga-o ao ambiente atual
        model = PPO.load(nome_modelo, env=env)
        
        # reset_num_timesteps=False faz com que os gráficos (logs) continuem 
        # a contar a partir de onde paraste (ex: passo 100.001)
        reset_timesteps = False
    else:
        print("--- Nenhum treino encontrado. A começar do ZERO ---")
        model = PPO(
            "CnnPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            device="cuda"
        )
        reset_timesteps = True

    print("--- A iniciar treino ---")
    
    # 3. Treinar
    # reset_num_timesteps=False é crucial para ele não "zerar" o contador de progresso interno
    model.learn(total_timesteps=100000, progress_bar=True, reset_num_timesteps=reset_timesteps)

    # 4. Guardar (Sobrescreve o ficheiro anterior com a versão mais inteligente)
    model.save(nome_modelo)
    print("Modelo atualizado e guardado.")
    
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