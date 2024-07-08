import numpy as np
from pendulum_env_extended import PendulumEnvExtended
import random
import wandb
import pickle

# Initialize Weights & Biases with a custom name
wandb.init(project="pendulum-qlearning_V2", name="pendulum_run_BigTraining")

# Define parameters
LEARNING_RATE = 0.1  # Alpha
DISCOUNT_FACTOR = 0.99  # Gamma
EPISODES = 5000  # K
TEST_EPISODES = 500 # L
ITERATIONS = 250  # N
EPSILON = 1.0  # Epsilon
EPSILON_DECAY = 0.995  # Decay rate
MIN_EPSILON = 0.01  # Minimum epsilon

# Discretize the action space and observation space
NUM_DISCRETE_ACTIONS = 10
NUM_DISCRETE_OBSERVATIONS = [20, 20, 200]

# Log parameters
wandb.config.update({
    "LEARNING_RATE": LEARNING_RATE,
    "DISCOUNT_FACTOR": DISCOUNT_FACTOR,
    "EPISODES": EPISODES,
    "TEST_EPISODES": TEST_EPISODES,
    "ITERATIONS": ITERATIONS,
    "EPSILON": EPSILON,
    "EPSILON_DECAY": EPSILON_DECAY,
    "MIN_EPSILON": MIN_EPSILON
})

# Create the environment
env = PendulumEnvExtended()

# Create bins for the observation space
obs_bins = [
    np.linspace(-1.0, 1.0, NUM_DISCRETE_OBSERVATIONS[0]),
    np.linspace(-1.0, 1.0, NUM_DISCRETE_OBSERVATIONS[1]),
    np.linspace(-8.0, 8.0, NUM_DISCRETE_OBSERVATIONS[2])
]

# Discretize the observation
def discretize_observation(observation):
    return tuple(
        int(np.digitize(observation[i], obs_bins[i]) - 1)
        for i in range(len(observation))
    )

# Initialize Q-table
q_table = np.random.uniform(low=-1, high=1, size=(NUM_DISCRETE_OBSERVATIONS + [NUM_DISCRETE_ACTIONS]))

# Function to choose action based on epsilon-greedy policy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, NUM_DISCRETE_ACTIONS - 1)
    else:
        return np.argmax(q_table[state])

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

# Training function
def train_agent(episodes, epsilon, epsilon_decay, min_epsilon):
    global q_table
    total_rewards = []
    total_steps = []
    
    for episode in range(episodes):
        observation, _ = env.reset()
        current_state = discretize_observation(observation)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = choose_action(current_state, epsilon)
            torque = np.linspace(-2, 2, NUM_DISCRETE_ACTIONS)[action]
            next_observation, reward, done, _, _ = env.step([torque])
            next_state = discretize_observation(next_observation)

            # Update Q-table using Q-learning algorithm
            best_next_action = np.argmax(q_table[next_state])
            q_table[current_state][action] = q_table[current_state][action] + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * q_table[next_state][best_next_action] - q_table[current_state][action]
            )

            current_state = next_state
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)
        total_steps.append(steps)

        # Log the results to wandb
        wandb.log({
            "train_episode": episode,
            "train_reward": total_reward,
            "train_steps": steps,
            "epsilon": epsilon
        })

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
    
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    
    return avg_reward, avg_steps

# Testing function
def test_agent(test_episodes):
    test_rewards = []
    test_steps = []
    
    for test_episode in range(test_episodes):
        observation, _ = env.reset()
        current_state = discretize_observation(observation)
        done = False
        total_test_reward = 0
        steps = 0

        while not done:
            action = np.argmax(q_table[current_state])
            torque = np.linspace(-2, 2, NUM_DISCRETE_ACTIONS)[action]
            next_observation, reward, done, _, _ = env.step([torque])
            next_state = discretize_observation(next_observation)

            current_state = next_state
            total_test_reward += reward
            steps += 1

        test_rewards.append(total_test_reward)
        test_steps.append(steps)
        
        # Log the results to wandb
        wandb.log({
            "test_episode": test_episode,
            "test_reward": total_test_reward,
            "test_steps": steps
        })
    
    avg_test_reward = np.mean(test_rewards)
    avg_test_steps = np.mean(test_steps)
    
    return avg_test_reward, avg_test_steps

# Run iterations of training and testing
for iteration in range(ITERATIONS):
    print(f"Starting iteration {iteration + 1}/{ITERATIONS}")
    avg_train_reward, avg_train_steps = train_agent(EPISODES, EPSILON, EPSILON_DECAY, MIN_EPSILON)
    avg_test_reward, avg_test_steps = test_agent(TEST_EPISODES)
    
    wandb.log({
        "iteration": iteration + 1,
        "avg_train_reward": avg_train_reward,
        "avg_train_steps": avg_train_steps,
        "avg_test_reward": avg_test_reward,
        "avg_test_steps": avg_test_steps
    })

# Save the final model
save_model(q_table, 'final_trained_model.pkl')

env.close()
wandb.finish()
