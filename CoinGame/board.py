import numpy as np
from tabulate import tabulate
import random

object = 1
empty_cell = 0

class Board:
    def __init__(self, board_size=(3, 3)):
        self.grid = self.create_tree_board(board_size[0])
        self.board_size = board_size
        
    def create_tree_board(self, n, p=.1):
        board = np.zeros((n, 2*n-1), dtype=int)
        for i in range(n):
            for j in range(n-i-1, n+i, 1):
                board[i, j] = random.random() > p if i != 0 else 1
        return board

    def play(self, action):
        """Realiza una jugada en el tablero"""
        row = action[0]
        from_col = action[1]
        to_col = action[2]
        
        if row < 0 or row >= self.board_size[0]:
            return False
        
        if self.grid[row, from_col:to_col+1].sum() != (to_col - from_col) + 1:
            return False
        
        self.grid[row, from_col:to_col+1] = 0
        
        return True
    
    def render_grid(self):
        """Imprime el tablero en consola"""
        num_rows, num_cols = self.board_size
        num_cols = num_cols * 2 - 1
        
        table = [["" for _ in range(num_cols + 1)] for _ in range(num_rows)]

        for row in range(num_rows):
            for col in range(num_cols):
                if self.grid[row, col] == object:
                    table[row][col + 1] = "O" 
                else:
                    table[row][col + 1] = " "
                    
        for i in range(num_rows):
            table[i][0] = str(i)

        headers = [str(col) for col in range(num_cols)]
        headers.insert(0, " ")
        table.insert(0, headers)
        
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    def clone(self):
        """Devuelve una copia del tablero"""
        board_clone = Board(board_size=self.board_size)
        board_clone.grid = np.copy(self.grid)
        return board_clone
    
    def is_end(self, player) -> tuple[bool, int]:
        """
        Devuelve: (True/False que indica si termino el juego, 1/2 que indica el jugador que ganó)
        """
        
        if self.grid.sum() <= 1:
            return True, player % 2 + 1
        return False, 0
    
    def get_possible_actions(self):
        """Devuelve una lista de acciones posibles para el jugador"""
        possible_actions = []
        for row in range(self.board_size[0]):
            from_col = 0
            for col in range(self.board_size[1] * 2 - 1):
                if self.grid[row, col] == object:
                    possible_actions.append((row, col, col))
                    from_col = col
                    for col2 in range(col + 1, self.board_size[1] * 2 - 1):
                        if self.grid[row, col2] == object:
                           possible_actions.append((row, from_col, col2))
                        else:
                            break
                        
        return possible_actions