from __future__ import annotations

import json
import os
import random

from battleship import (
    BayesianStrategy,
    HiddenBoard,
    HuntTargetStrategy,
    ObservationState,
    RandomStrategy,
    play_game,
    posterior_heatmap,
    random_board,
    render_heatmap_svg,
    render_histogram_svg,
    summarize,
)


OUTPUT_DIR = "outputs"


def make_outputs_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_example_visuals():
    rng = random.Random(2026)
    board = random_board(rng)
    strategy = BayesianStrategy(sample_count=180)
    state = ObservationState()
    history = []

    for _ in range(8):
        shot, info = strategy.choose_shot(state, rng)
        result, sunk_ship = board.fire(shot)
        state.register_shot(shot, result, sunk_ship)
        history.append((shot, result))
        if result == "hit":
            break

    heat, used = posterior_heatmap(state, 260, rng)

    render_heatmap_svg(
        heat,
        os.path.join(OUTPUT_DIR, "example_heatmap.svg"),
        title="Posterior probability heatmap after first evidence",
        best_cell=max(
            state.unknown_cells(),
            key=lambda cell: (heat[cell[0]][cell[1]], -abs(cell[0] - 4.5) - abs(cell[1] - 4.5)),
        ),
        shots=history,
        true_board=board,
    )

    return {
        "history": history,
        "counted_placements": used,
    }


def simulate_games():
    seed = 314159
    board_rng = random.Random(seed)

    bayes_strategy = BayesianStrategy(sample_count=120)
    hunt_target_strategy = HuntTargetStrategy()
    random_strategy = RandomStrategy()

    bayes_shots = []
    hunt_target_shots = []
    random_shots = []

    ship_layouts = [random_board(board_rng).ships for _ in range(80)]

    play_rng_bayes = random.Random(seed + 1)
    play_rng_hunt = random.Random(seed + 2)
    play_rng_random = random.Random(seed + 3)

    for ships in ship_layouts:
        board = HiddenBoard(ships)
        result = play_game(bayes_strategy, board, play_rng_bayes)
        bayes_shots.append(result["shots"])

    for ships in ship_layouts:
        board = HiddenBoard(ships)
        result = play_game(hunt_target_strategy, board, play_rng_hunt)
        hunt_target_shots.append(result["shots"])

    for ships in ship_layouts:
        board = HiddenBoard(ships)
        result = play_game(random_strategy, board, play_rng_random)
        random_shots.append(result["shots"])

    render_histogram_svg(
        [("Bayes", bayes_shots), ("Hunt-target", hunt_target_shots), ("Random", random_shots)],
        os.path.join(OUTPUT_DIR, "shot_distribution.svg"),
        title="Distribution of shots needed to finish the game",
    )

    return {
        "bayes": summarize(bayes_shots),
        "hunt_target": summarize(hunt_target_shots),
        "random": summarize(random_shots),
        "bayes_shots": bayes_shots,
        "hunt_target_shots": hunt_target_shots,
        "random_shots": random_shots,
    }


def main():
    make_outputs_dir()
    visuals = make_example_visuals()
    simulations = simulate_games()

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "visuals": visuals,
                "simulations": simulations,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved outputs to outputs/")
    print(json.dumps(simulations, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
