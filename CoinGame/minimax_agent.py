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
                return -1  # Perder es malo
            else:
                return 1  # Ganar es bueno

        # Caso 2: Contar el número de monedas restantes
        total_coins = sum(1 for row in range(board.board_size[0]) for col in range(board.board_size[1] * 2 - 1) if board.grid[row, col] == object)
        
        # Si queda solo una moneda, es una buena posición si es el turno del oponente
        if total_coins == 1:
            return 1 if player != self.player else -1
        
        # Caso 3: Paridad de filas
        odd_rows = sum(1 for row in range(board.board_size[0]) if sum(1 for col in range(board.board_size[1] * 2 - 1) if board.grid[row, col] == object) % 2 != 0)
        
        if odd_rows % 2 != 0:
            return 1  # Bueno tener un número impar de filas con un número impar de monedas
        else:
            return -1  # Malo tener un número par de filas con un número impar de monedas
