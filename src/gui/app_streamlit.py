import streamlit as st
import chess

from src.engine.engine_wrapper import EngineWrapper
from src.openings.opening_explorer import OpeningExplorer
from src.trainer.eval_bar import render_eval_bar

st.set_page_config(page_title="Chess Openings Trainer", layout="wide")
st.title("♟️ Chess Openings Trainer")

engine_path = st.text_input("Engine path", value="stockfish")
depth = st.slider("Depth", 8, 30, 16)
multipv = st.slider("MultiPV", 1, 5, 3)

col1, col2 = st.columns([2,1])
with col1:
    fen = st.text_input("FEN", value=chess.STARTING_FEN)
    board = chess.Board(fen)
    explorer = OpeningExplorer()
    st.write("**Opening:**", (lambda m: f"{m.name} (ECO {m.eco}) – {m.matched_moves} moves" if (m:=explorer.identify(board)) else "[unknown]")())

with col2:
    with EngineWrapper(engine_path, depth, multipv) as eng:
        lines = eng.analyse(board)
        if lines:
            bar, label = render_eval_bar(lines[0].cp, lines[0].mate)
            st.write(bar, label)
            st.write("**Top options:**")
            for i, ln in enumerate(lines[:3], 1):
                st.write(f"{i}) {board.san(ln.move)} — {ln.pv_san}  ({('#'+str(abs(ln.mate))) if ln.mate is not None else f'{(ln.cp or 0)/100:+.2f}'})")
        else:
            st.write("_No engine output._")
