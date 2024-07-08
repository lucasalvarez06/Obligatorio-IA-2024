from gymnasium.envs.classic_control.pendulum import PendulumEnv
import numpy as np

class PendulumEnvExtended(PendulumEnv):
    def __init__(self, *args, **kwargs):
        super(PendulumEnvExtended, self).__init__(*args, **kwargs)
        self.max_steps = 700
        self.actual_steps = 0
        
    def step(self, action):
        self.actual_steps += 1
        obs, reward, _, _, _ = super().step(action)
        # Check if the pendulum is sufficiently close to the upright position
        theta = np.arccos(obs[0])  # obs[0] = cos(theta), so theta = arccos(cos(theta))
        close_to_upright = abs(theta) < 0.1  # Threshold for considering it "upright"
        
        done = self.actual_steps >= self.max_steps or close_to_upright
        return obs, reward, done, False, {}
    
    def reset(self, *args, **kwargs):
        self.actual_steps = 0
        return super().reset(*args, **kwargs)