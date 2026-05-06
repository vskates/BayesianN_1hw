import random
import unittest

from battleship import BOARD_SIZE, BayesianStrategy, FLEET, ObservationState, posterior_heatmap, random_board


class BattleshipTests(unittest.TestCase):
    def test_random_board(self):
        board = random_board(random.Random(1))
        self.assertEqual(len(board.ship_cells), sum(FLEET))

        ships = [set(ship) for ship in board.ships]
        for i in range(len(ships)):
            for j in range(i + 1, len(ships)):
                for r, c in ships[i]:
                    for rr in range(r - 1, r + 2):
                        for cc in range(c - 1, c + 2):
                            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                                self.assertNotIn((rr, cc), ships[j])

    def test_heatmap_and_shot(self):
        state = ObservationState()
        heat, used = posterior_heatmap(state)
        self.assertGreater(used, 0)

        best = None
        score = None
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if score is None or heat[r][c] > score:
                    score = heat[r][c]
                    best = (r, c)
        self.assertTrue(3 <= best[0] <= 6 and 3 <= best[1] <= 6)

        shot, _ = BayesianStrategy().choose_shot(state, random.Random(4))
        self.assertNotIn(shot, state.fired_cells())


if __name__ == "__main__":
    unittest.main()
