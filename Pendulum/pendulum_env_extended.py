from gymnasium.envs.classic_control.pendulum import PendulumEnv

class PendulumEnvExtended(PendulumEnv):
    def __init__(self, *args, **kwargs):
        super(PendulumEnvExtended, self).__init__(*args, **kwargs)
        self.max_steps = 700
        self.actual_steps = 0
        
    def step(self, action):
        self.actual_steps += 1
        obs, reward, _, _, _ = super().step(action)
        done = self.actual_steps >= self.max_steps # asegurarnos que el episodio no dure más de 700 pasos
        return obs, reward, done, False, {}
    
    def reset(self, *args, **kwargs):
        self.actual_steps = 0
        return super().reset(*args, **kwargs)