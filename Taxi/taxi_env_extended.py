from gymnasium.envs.toy_text.taxi import TaxiEnv
class TaxiEnvExtended(TaxiEnv):
    def __init__(self, render_mode="rgb_array", *args, **kwargs):
        super(TaxiEnvExtended, self).__init__(render_mode=render_mode, *args, **kwargs)
        self.max_steps = 200
        self.actual_steps = 0
        
    def step(self, action):
        self.actual_steps += 1
        obs, reward, done, _, _ = super().step(action)
        done = done or self.actual_steps >= self.max_steps # asegurarnos que el episodio no dure más de 200 pasos
        return obs, reward, done, False, {}
    
    def reset(self, *args, **kwargs):
        self.actual_steps = 0
        return super().reset(*args, **kwargs)
    
    