#!/usr/bin/env bash
python -m src.trainer.trainer_cli --engine "${ENGINE_PATH:-$(which stockfish)}" --depth "${ENGINE_DEPTH:-16}" --side "${1:-white}" --mode "${2:-drill}"
