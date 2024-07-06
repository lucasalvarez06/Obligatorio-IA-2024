# -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym
from taxi_env_extended import TaxiEnvExtended
import wandb

def optimal_policy(state, Q):
    return np.argmax(Q[state])

def epsilon_greedy_policy(state, Q, env, epsilon=0.1):
    explore = np.random.binomial(1, epsilon)
    if explore:
        action = env.action_space.sample()
        # print('explore')
    else:
        action = np.argmax(Q[state])
        # print('exploit')
        
    return action

def train(env, Q, K, alpha, gamma, epsilon):
    for episode in range(K):
        obs, _ = env.reset()
        done = False
        
        while not done:
            state = obs
            action = epsilon_greedy_policy(state, Q, env, epsilon)
            obs, reward, done, _, _ = env.step(action)
            
            # Actualizar la tabla Q
            next_state = obs
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_delta = td_target - Q[state, action]
            Q[state, action] += alpha * td_delta

def test(env, Q, L):
    total_rewards = []
    total_steps = []

    for _ in range(L):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            state = obs
            action = optimal_policy(state, Q)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1
        
        total_rewards.append(total_reward)
        total_steps.append(step_count)

    return np.mean(total_rewards), np.mean(total_steps)

def main(K, N, L, alpha, gamma, epsilon):
    # Inicializar wandb
    wandb.init(project="taxi-v3-q-learning", config={
        "K": K,
        "N": N,
        "L": L,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon
    })

    # Crear el entorno
    env = TaxiEnvExtended()
    actions = env.action_space.n
    states = env.observation_space.n

    # Inicializar la tabla Q con ceros
    Q = np.zeros((states, actions))

    # Iterar
    for i in range(N):
        # Ajustar epsilon
        epsilon = max(0.01, epsilon * 0.99)
        print("Iteration {}/{}, Epsilon: {}".format(i+1, N, epsilon))

        # Entrenar
        train(env, Q, K, alpha, gamma, epsilon)

        # Probar
        avg_reward, avg_steps = test(env, Q, L)
        print("Average Reward: {}, Average Steps: {}".format(avg_reward, avg_steps))

        # Registrar los resultados en wandb
        wandb.log({"Iteration": i + 1, "Epsilon": epsilon, "Average Reward": avg_reward, "Average Steps": avg_steps})


if __name__ == "__main__":
    K = 1000  # Cantidad de episodios de entrenamiento por iteración
    N = 10  # Cantidad de iteraciones
    L = 100  # Cantidad de episodios de prueba por iteración
    alpha = 0.1  # Tasa de aprendizaje
    gamma = 0.99  # Factor de descuento
    epsilon = 0.1  # Probabilidad de exploración inicial

    main(K, N, L, alpha, gamma, epsilon)
