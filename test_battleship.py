import random
import unittest

from battleship import (
    BOARD_SIZE,
    BayesianStrategy,
    FLEET,
    ObservationState,
    posterior_heatmap,
    random_board,
)


class BattleshipTests(unittest.TestCase):
    def test_random_board_has_correct_ship_cells(self):
        rng = random.Random(1)
        board = random_board(rng)
        self.assertEqual(len(board.ship_cells), sum(FLEET))

    def test_random_board_ships_do_not_touch(self):
        rng = random.Random(2)
        board = random_board(rng)
        ships = [set(ship) for ship in board.ships]

        for i in range(len(ships)):
            for j in range(i + 1, len(ships)):
                for r, c in ships[i]:
                    for rr in range(r - 1, r + 2):
                        for cc in range(c - 1, c + 2):
                            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                                self.assertNotIn((rr, cc), ships[j])

    def test_empty_board_heatmap_prefers_center(self):
        rng = random.Random(3)
        state = ObservationState()
        heat, used = posterior_heatmap(state, 80, rng)
        self.assertGreater(used, 0)

        best = None
        best_score = None
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                score = heat[r][c]
                if best_score is None or score > best_score:
                    best_score = score
                    best = (r, c)

        self.assertTrue(3 <= best[0] <= 6)
        self.assertTrue(3 <= best[1] <= 6)

    def test_bayesian_strategy_returns_unknown_cell(self):
        rng = random.Random(4)
        state = ObservationState()
        strategy = BayesianStrategy(sample_count=40)
        shot, _ = strategy.choose_shot(state, rng)
        self.assertNotIn(shot, state.fired_cells())


if __name__ == "__main__":
    unittest.main()
