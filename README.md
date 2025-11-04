# ♟️ Chess Openings Trainer

An interactive, Stockfish-powered chess openings trainer that:

- Recognizes and names openings (ECO codes & move sequences)
- Shows best moves, candidate lines, and live evaluation bars
- Lets you practice openings (drill mode), play freely (free mode), or test blindfolded
- Learns from your play using a lightweight PyTorch neural network (value + policy)
- Runs fully in the terminal, with an optional Streamlit GUI

## 🚀 Quick start

### 1. Clone & install
```bash
git clone https://github.com/yourusername/chess-openings-trainer.git
cd chess-openings-trainer
pip install -r requirements.txt

