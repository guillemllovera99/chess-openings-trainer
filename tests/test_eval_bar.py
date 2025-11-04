from src.trainer.eval_bar import render_eval_bar

def test_eval_bar_shapes():
    bar, label = render_eval_bar(50.0, None)
    assert isinstance(bar, str) and isinstance(label, str)
