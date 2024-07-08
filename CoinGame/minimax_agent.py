from agent import Agent

class MinimaxAgent(Agent):
    def __init__(self, player, max_depth=4):
        super().__init__(player)
        self.max_depth = max_depth

    def next_action(self, obs):
        def minimax(board, depth, is_maximizing_player):
            is_end, winner = board.is_end(self.player)
            if is_end or depth == self.max_depth:
                return self.heuristic_utility(board, self.player)
            
            if is_maximizing_player:
                max_eval = float('-inf')
                for move in board.get_possible_actions():
                    new_board = board.clone()
                    new_board.play(move)
                    eval = minimax(new_board, depth + 1, False)
                    max_eval = max(max_eval, eval)
                return max_eval
            else:
                min_eval = float('inf')
                for move in board.get_possible_actions():
                    new_board = board.clone()
                    new_board.play(move)
                    eval = minimax(new_board, depth + 1, True)
                    min_eval = min(min_eval, eval)
                return min_eval

        best_value = float('-inf')
        best_move = None
        for move in obs.get_possible_actions():
            new_board = obs.clone()
            new_board.play(move)
            move_value = minimax(new_board, 0, False)
            if move_value > best_value:
                best_value = move_value
                best_move = move
        return best_move

    def heuristic_utility(self, board, player):
        opponent = (player % 2) + 1
        is_end, winner = board.is_end(player)
        
        # Caso 1: Estado terminal
        if is_end:
            if winner == player:
                return 1  # Ganar es bueno
            else:
                return -1  # Perder es malo

        # Caso 2: Contar el número de monedas restantes
        total_coins = sum(1 for row in range(board.board_size[0]) for col in range(board.board_size[1] * 2 - 1) if board.grid[row, col] == 1)
        
        # Caso 3: Paridad de filas
        odd_rows = sum(1 for row in range(board.board_size[0]) if sum(1 for col in range(board.board_size[1] * 2 - 1) if board.grid[row, col] == 1) % 2 != 0)
        
        # Evaluar el Nim-Sum
        nim_sum = 0
        for row in range(board.board_size[0]):
            row_sum = sum(1 for col in range(board.board_size[1] * 2 - 1) if board.grid[row, col] == 1)
            nim_sum ^= row_sum
        
        # Heurística mejorada
        utility = 0
        
        # Factor 1: Maximizar el número de monedas propias restantes
        utility += total_coins
        
        # Factor 2: Minimizar el número de filas impares (dejar al oponente en desventaja)
        utility -= odd_rows
        
        # Factor 3: Maximizar posiciones donde el Nim-Sum no es cero
        if nim_sum != 0:
            utility += 1
        else:
            utility -= 1
        
        # Ajustar el valor final de la heurística
        return utility
