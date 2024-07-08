# -*- coding: utf-8 -*-

import pickle
import numpy as np
import gymnasium as gym
from taxi_env_extended import TaxiEnvExtended
import wandb

def optimal_policy(state, Q):
    return np.argmax(Q[state])

def epsilon_greedy_policy(state, Q, env, epsilon):
    explore = np.random.binomial(1, epsilon)
    if explore:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
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


# Function to save the model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Modelo guardado en {filename}")
 
# Function to load the model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Modelo cargado desde {filename}")
    return model

def main(K, N, L, alpha_start, alpha_end, alpha_decay, gamma, epsilon_start, epsilon_end, epsilon_decay):
    # Inicializar wandb
    wandb.init(project="taxi-v3-q-learning", config={
        "K": K,
        "N": N,
        "L": L,
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "alpha_decay": alpha_decay,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay
    })

    # Crear el entorno
    env = TaxiEnvExtended()
    actions = env.action_space.n
    states = env.observation_space.n

    # Inicializar la tabla Q con ceros
    Q = np.zeros((states, actions))

    alpha = alpha_start
    epsilon = epsilon_start

    # Iterar
    for i in range(N):
        # Entrenar
        train(env, Q, K, alpha, gamma, epsilon)

        # Probar
        avg_reward, avg_steps = test(env, Q, L)
        print("Iteration {}/{}, Epsilon: {}, Alpha: {}, Average Reward: {}, Average Steps: {}".format(
            i + 1, N, epsilon, alpha, avg_reward, avg_steps))

        # Registrar los resultados en wandb
        wandb.log({"Iteration": i + 1, "Epsilon": epsilon, "Alpha": alpha, "Average Reward": avg_reward, "Average Steps": avg_steps})

        # Ajustar epsilon y alpha
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        alpha = max(alpha_end, alpha * alpha_decay)

    save_model(Q, "final_taxi_trained_model_2.pkl")


if __name__ == "__main__":
    K = 2000  # Cantidad de episodios de entrenamiento por iteración
    N = 200  # Cantidad de iteraciones
    L = 10  # Cantidad de episodios de prueba por iteración
    gamma = 0.99  # Factor de descuento
    epsilon_start = 0.9  # Valor inicial de epsilon
    epsilon_end = 0.1  # Valor final de epsilon
    epsilon_decay = 0.95  # Factor de decaimiento de epsilon
    alpha_start = 0.1  # Valor inicial de alpha
    alpha_end = 0.01  # Valor final de alpha
    alpha_decay = 0.95  # Factor de decaimiento de alpha

    main(K, N, L, alpha_start, alpha_end, alpha_decay, gamma, epsilon_start, epsilon_end, epsilon_decay)
