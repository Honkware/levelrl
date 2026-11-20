# chess-eval

The evaluation code from LevelRL, split out so it can be used with any chess policy. It scores a policy on positions where the human move is known, and it estimates a rating from games against fixed opponents. It doesn't import Maia or LevelRL, and the only dependency is `python-chess`.

```bash
python -m pip install ./chess_eval
```

A policy is any object with a `predict(position)` method that returns probabilities keyed by legal UCI moves.

```python
from chess_eval import ProbePosition, evaluate_probe_set, summarize

class UniformPredictor:
    def predict(self, position):
        moves = list(position.board().legal_moves)
        return {move.uci(): 1 / len(moves) for move in moves}

positions = [ProbePosition.from_moves([], "e2e4")]
scores = evaluate_probe_set(positions, UniformPredictor())
print(summarize(scores))  # {'n': 1, 'top1': 0.0, 'log_loss': ...}
```

`top1` is how often the policy's first choice is the move the person played, and `log_loss` is the negative log probability it gave that move. `paired_ci(a, b)` gives a bootstrap interval for the difference between two policies on the same positions, and `load_probe_set` reads positions from JSONL.

For strength, `estimate_rating` computes a maximum-likelihood Elo from results against opponents of known rating, and `bootstrap_rating` adds an interval by resampling games. `fit_rating_scale` fits a conversion between two rating scales, the same way LevelRL turns ladder ratings into Lichess ratings.

`analyse_move` works out centipawn loss for a move given an engine adapter, so it isn't tied to Stockfish.

MIT licensed.
