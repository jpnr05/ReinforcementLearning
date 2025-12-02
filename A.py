import gymnasium as gym
import ale_py  # <--- Isto é CRÍTICO na nova versão

print(f"Gymnasium versão: {gym.__version__}")
print(f"ALE-Py versão: {ale_py.__version__}")

# Força o registo dos ambientes Atari
gym.register_envs(ale_py)

# Tenta encontrar ambientes Atari
all_envs = list(gym.envs.registry.keys())
atari_envs = [env for env in all_envs if "ALE/" in env]

if len(atari_envs) > 0:
    print(f"\n✅ SUCESSO! {len(atari_envs)} jogos Atari encontrados.")
    print(f"Exemplo: {atari_envs[0]}")
    
    # Teste rápido do Pong
    try:
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        env.reset()
        print("✅ Ambiente Pong carregado e pronto a jogar!")
        env.close()
    except Exception as e:
        print(f"Erro ao carregar o Pong: {e}")
else:
    print("\n❌ AVISO: Ainda não foram encontrados ambientes Atari.")
    print("Certifica-te que correste o comando 'AutoROM --accept-license' no terminal.")