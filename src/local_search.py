from ai_agent import evaluate_game_state
import time
import random

def get_best_move_local_search(game, max_time=5):
    """
    Bot 4: Menggunakan algoritma Hill Climbing.
    """
    start_time = time.time()
    best_move = None
    best_score = evaluate_game_state(game)

    valid_moves = game.get_valid_moves()
    if not valid_moves:
        return None

    for move in valid_moves:
        new_state = game.copy()
        new_state.make_move(*move)
        new_score = evaluate_game_state(new_state)

        if new_score >= best_score:
            best_score = new_score
            best_move = move

        if time.time() - start_time >= max_time:
            break
    
    if best_move == None:
        best_move = random.choice(valid_moves)

    return best_move
