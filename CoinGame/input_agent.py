from agent import Agent
from board import Board

class InputAgent(Agent):
    
    def __init__(self, player=1):
        super().__init__(player)
        
    def next_action(self, obs):
        while True:
            try:
                row = int(input("Row: "))
                from_col = int(input("From column: "))
                to_col = int(input("To column: "))
                input_value = (row, from_col, to_col)
                return input_value
            except ValueError:
                print("Please insert a valid action.")

    def heuristic_utility(self, board: Board):
        return 0