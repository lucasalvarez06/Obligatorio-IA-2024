from datetime import datetime
from board import Board as GameBoard
from random_agent import RandomAgent
from agent import Agent
from coin_game_env import CoinGameEnv

def play_vs_other_agent(env, agent1, agent2, render = False):
    done = False
    obs = env.reset()
    winner = 0
    while not done:
        if render: env.render()
        action = agent1.next_action(obs)
        obs, _, done, winner, _ = env.step(action)

        if render: env.render()
        if not done:
            next_action = agent2.next_action(obs)
            _, _, done, winner, _ = env.step(next_action)

    if render: env.render()
    
    return winner