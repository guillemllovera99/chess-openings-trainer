from __future__ import annotations
import sys
from typing import List

import chess
import chess.pgn

from src.config import parse_args
from src.engine.engine_wrapper import EngineWrapper
from src.openings.opening_explorer import OpeningExplorer
from src.trainer.eval_bar import render_eval_bar
from src.trainer.scoring import cp_loss, verdict

from src.learning.features import board_to_planes, targets_from_engine
from src.learning.learner import OnlineLearner


def _opening_string(explorer: OpeningExplorer, board: chess.Board) -> str:
    m = explorer.identify(board)
    if not m:
        return "Opening: [unknown]"
    return f"Opening: {m.name} (ECO {m.eco}) – matched {m.matched_moves} move(s)."


def main() -> int:
    engine_cfg, trainer_cfg = parse_args()
    explorer = OpeningExplorer()
    learner = OnlineLearner()   # <-- NEW

    board = chess.Board()
    user_is_white = trainer_cfg.side.lower() == "white"

    print("=== Chess Openings Trainer (Learning Enabled) ===")

    with EngineWrapper(engine_cfg.engine_path, engine_cfg.depth, engine_cfg.multipv) as eng:
        while not board.is_game_over():

            # ENGINE MOVE WHEN NOT USER TURN
            if (board.turn and not user_is_white) or ((not board.turn) and user_is_white):
                lines = eng.analyse(board)
                if not lines:
                    break
                best = lines[0]
                engine_san = board.san(best.move)
                board.push(best.move)

                bar, label = render_eval_bar(best.cp, best.mate)
                print(f"Engine: {engine_san}   {bar}   {label}")
                print(_opening_string(explorer, board))
                continue

            # USER TURN
            print("\nFEN:", board.fen())
            print(_opening_string(explorer, board))
            lines = eng.analyse(board)

            cmd = input("Your move (SAN/hint/best/undo/quit): ").strip().lower()

            if cmd in ("quit", "q"):
                break
            if cmd == "undo":
                if board.move_stack: board.pop()
                if board.move_stack: board.pop()
                continue
            if cmd == "hint":
                for i, ln in enumerate(lines[:3], 1):
                    print(i, board.san(ln.move), ln.pv_san)
                continue
            if cmd == "best":
                ln = lines[0]
                print("Best:", board.san(ln.move), ln.pv_san)
                continue

            try:
                move = board.parse_san(cmd)
            except:
                print("Invalid SAN.")
                continue

            # LEARNING HOOK — before move is made
            best_before = lines[0]
            best_cp_before = best_before.cp
            best_move_before = best_before.move

            user_san = board.san(move)
            board.push(move)

            # Quick position eval after user move for scoring + training target
            after = eng.analyse(board)[0]
            user_cp_after = after.cp

            # Score output
            is_best = (move == best_move_before)
            loss = cp_loss(best_cp_before, user_cp_after)
            print(f"{verdict(is_best, loss)} — You played {user_san}"
                  + (f"; Δcp ≈ {loss:.0f}" if loss is not None else ""))

            # **LEARN FROM THIS POSITION**
            x = board_to_planes(board)
            value_target, policy_index = targets_from_engine(user_cp_after)
            learner.step_after_move(x, value_target, policy_index, updates=2)

            # Show eval bar
            bar, label = render_eval_bar(after.cp, after.mate)
            print(bar, label)

        print("\nSession complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
