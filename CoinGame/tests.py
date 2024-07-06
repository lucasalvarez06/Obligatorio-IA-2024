from coin_game_env import CoinGameEnv
from minimax_agent import MinimaxAgent
from random_agent import RandomAgent
from play import play_vs_other_agent

def test_agents(num_games):
    env = CoinGameEnv()
    minimax_agent = MinimaxAgent(1)
    random_agent = RandomAgent(2)
    
    minimax_wins = 0
    random_wins = 0
    
    for _ in range(num_games):
        env.reset()
        result = play_vs_other_agent(env, agent1=minimax_agent, agent2=random_agent, render=False)
        if result == 1:
            minimax_wins += 1
        else:
            random_wins += 1
    
    print(f"MinimaxAgent wins: {minimax_wins}")
    print(f"RandomAgent wins: {random_wins}")
