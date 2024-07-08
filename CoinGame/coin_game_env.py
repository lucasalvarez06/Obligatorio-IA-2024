import random
import gymnasium as gym
import numpy as np
from board import Board

player_1 = 1
player_2 = 2

class CoinGameEnv(gym.Env):
    def __init__(self, grid_size=3):
        super(CoinGameEnv, self).__init__()
        self.grid_size = grid_size
        #self.action_space = spaces.Discrete(grid_size*(grid_size+1)//2)
        #self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size,grid_size), dtype=np.float32)
        self.reset()
        
    def reset(self):
        self.current_player = player_1
        self.grid = Board((self.grid_size, self.grid_size))
        return self.grid

    def step(self, action):
        action_done = self.grid.play(action)
        
        if action_done:
            self.next_player()
        else:
            raise ValueError("Invalid action")
            
        done, winner = self.is_done()
        return self.grid, self.get_reward(), done, winner, {}
    
    def is_done(self):        
        return self.grid.is_end(self.current_player)
    
    def get_reward(self):
        return 1 if self.current_player == player_1 else -1   
        
    def next_player(self):
        self.current_player = player_1 if self.current_player == player_2 \
                                    else player_2
    def render(self):
        self.grid.render_grid()



