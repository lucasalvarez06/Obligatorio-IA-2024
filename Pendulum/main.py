import gymnasium as gym
import numpy as np
import random
import wandb
from pendulum_env_extended import PendulumEnvExtended

def optimal_policy(state, Q):
    return np.argmax(Q[state])

def epsilon_greedy_policy(state, Q, env, epsilon=0.1):
    explore = np.random.binomial(1, epsilon)
    if explore:
        action = random.choice([-2, 0, 2])  # Exploración
    else:
        action = np.argmax(Q[state])
    return action

def discretize_state(state, state_bins):
    cos_theta, sin_theta, theta_dot = state
    indices = []
    for i, val in enumerate([cos_theta, sin_theta, theta_dot]):
        index = np.digitize(val, state_bins[i]) - 1
        index = min(index, len(state_bins[i]) - 1)
        index = max(index, 0)
        indices.append(index)
    return tuple(indices)

def train(env, Q, K, alpha, gamma, epsilon, state_bins):
    for episode in range(K):
        obs, _ = env.reset()
        done = False
        
        while not done:
            state = discretize_state(obs, state_bins)
            action = epsilon_greedy_policy(state, Q, env, epsilon)
            real_action = np.array([action])
            obs, reward, done, _, _ = env.step(real_action)
            
            next_state = discretize_state(obs, state_bins)
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_delta = td_target - Q[state, action]
            Q[state, action] += alpha * td_delta

def test(env, Q, L, state_bins):
    total_rewards = []
    total_steps = []

    for _ in range(L):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            state = discretize_state(obs, state_bins)
            action = optimal_policy(state, Q)
            real_action = np.array([action])
            obs, reward, done, _, _ = env.step(real_action)
            total_reward += reward
            step_count += 1
        
        total_rewards.append(total_reward)
        total_steps.append(step_count)

    return np.mean(total_rewards), np.mean(total_steps)

def main(K, N, L, alpha, gamma, epsilon):
    # Inicializar wandb
    wandb.init(project="pendulum-q-learning", config={
        "K": K,
        "N": N,
        "L": L,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon
    })

    # Crear el entorno
    env = PendulumEnvExtended()
    action_size = 3  # Número de acciones posibles
    state_bins = [
        np.linspace(-1, 1, 50),  # cos(theta)
        np.linspace(-1, 1, 50),  # sin(theta)
        np.linspace(-8, 8, 50)   # theta_dot
    ]

    # Inicializar la tabla Q con ceros
    Q = np.zeros(state_bins[0].size * state_bins[1].size * state_bins[2].size * action_size).reshape(state_bins[0].size, state_bins[1].size, state_bins[2].size, action_size)

    # Iterar
    for i in range(N):
        # Ajustar epsilon
        epsilon = max(0.01, epsilon * 0.99)
        print(f"Iteration {i + 1}/{N}, Epsilon: {epsilon}")

        # Entrenar
        train(env, Q, K, alpha, gamma, epsilon, state_bins)

        # Probar
        avg_reward, avg_steps = test(env, Q, L, state_bins)
        print(f"Average Reward: {avg_reward}, Average Steps: {avg_steps}")

        # Registrar los resultados en wandb
        wandb.log({"Iteration": i + 1, "Epsilon": epsilon, "Average Reward": avg_reward, "Average Steps": avg_steps})

if __name__ == "__main__":
    K = 1000  # Cantidad de episodios de entrenamiento por iteración
    N = 10  # Cantidad de iteraciones
    L = 100  # Cantidad de episodios de prueba por iteración
    alpha = 0.1  # Tasa de aprendizaje
    gamma = 0.95  # Factor de descuento
    epsilon = 0.5  # Probabilidad de exploración inicial

    main(K, N, L, alpha, gamma, epsilon)
