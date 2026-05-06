import json
import os
import random

from battleship import (
    BayesianStrategy,
    HiddenBoard,
    ObservationState,
    RandomStrategy,
    play_game,
    posterior_heatmap,
    random_board,
    render_heatmap_svg,
    render_histogram_svg,
    summarize,
)


OUT = "outputs"


def make_example():
    rng = random.Random(2026)
    board = random_board(rng)
    state = ObservationState()
    strategy = BayesianStrategy()
    history = []

    while not history or history[-1][1] == "miss":
        shot, _ = strategy.choose_shot(state, rng)
        result, sunk = board.fire(shot)
        state.register_shot(shot, result, sunk)
        history.append((shot, result))

    heat, total = posterior_heatmap(state)
    best = max(state.unknown_cells(), key=lambda x: (heat[x[0]][x[1]], -abs(x[0] - 4.5) - abs(x[1] - 4.5)))
    render_heatmap_svg(heat, os.path.join(OUT, "example_heatmap.svg"), "Posterior probability heatmap", best, history, board)
    return {"history": history, "counted_placements": total}


def simulate():
    seed = 314159
    board_rng = random.Random(seed)
    boards = [random_board(board_rng).ships for _ in range(80)]
    strategies = [
        ("bayes", BayesianStrategy(), random.Random(seed + 1)),
        ("random", RandomStrategy(), random.Random(seed + 2)),
    ]
    res = {}

    for name, strategy, rng in strategies:
        shots = []
        for ships in boards:
            shots.append(play_game(strategy, HiddenBoard(ships), rng)["shots"])
        res[name] = summarize(shots)
        res[name + "_shots"] = shots

    render_histogram_svg(
        [("Bayes", res["bayes_shots"]), ("Random", res["random_shots"])],
        os.path.join(OUT, "shot_distribution.svg"),
        "Distribution of shots needed to finish the game",
    )
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {"visuals": make_example(), "simulations": simulate()}

    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Saved outputs to outputs/")
    print(json.dumps(data["simulations"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
