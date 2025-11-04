import os
import argparse
from dataclasses import dataclass

@dataclass
class EngineConfig:
    engine_path: str
    depth: int
    multipv: int

@dataclass
class TrainerConfig:
    side: str
    mode: str
    eco_pgn: str | None

def load_env_default(key: str, fallback):
    val = os.environ.get(key)
    return val if val is not None else fallback

def parse_args():
    parser = argparse.ArgumentParser(description="Chess Openings Trainer")
    parser.add_argument(
        "--engine", type=str,
        default=load_env_default("ENGINE_PATH", "stockfish"),
        help="Path to Stockfish or UCI engine binary"
    )
    parser.add_argument(
        "--depth", type=int,
        default=int(load_env_default("ENGINE_DEPTH", 16)),
        help="Search depth (engine strength)"
    )
    parser.add_argument(
        "--multipv", type=int,
        default=int(load_env_default("ENGINE_MULTIPV", 3)),
        help="Number of best move candidates to show"
    )
    parser.add_argument(
        "--side", type=str, choices=["white", "black"], default="white",
        help="Which side the human plays"
    )
    parser.add_argument(
        "--mode", type=str, choices=["drill", "blind", "free"], default="drill",
        help="Training mode"
    )
    parser.add_argument(
        "--eco-pgn", type=str, default=None,
        help="Optional ECO PGN to expand opening recognition"
    )

    args = parser.parse_args()

    engine_cfg = EngineConfig(
        engine_path=args.engine,
        depth=args.depth,
        multipv=args.multipv
    )

    trainer_cfg = TrainerConfig(
        side=args.side,
        mode=args.mode,
        eco_pgn=args.eco_pgn
    )

    return engine_cfg, trainer_cfg
