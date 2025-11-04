import chess
from src.openings.opening_explorer import OpeningExplorer

def test_ruy_lopez_detection():
    ex = OpeningExplorer()
    b = chess.Board()
    for san in ["e4","e5","Nf3","Nc6","Bb5"]:
        b.push_san(san)
    m = ex.identify(b)
    assert m and m.eco == "C60" and "Ruy Lopez" in m.name
