from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass


BOARD_SIZE = 10
FLEET = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
Cell = tuple[int, int]


def inside(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def neighbors8(cell):
    r, c = cell
    out = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if inside(rr, cc):
                out.add((rr, cc))
    return out


def neighbors4(cell):
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if inside(rr, cc):
            out.append((rr, cc))
    return out


@dataclass(frozen=True)
class Placement:
    length: int
    cells: tuple[Cell, ...]
    orientation: str
    cell_set: frozenset[Cell]
    halo: frozenset[Cell]


def make_placement(r, c, length, orientation):
    cells = []
    for k in range(length):
        if orientation == "H":
            cells.append((r, c + k))
        else:
            cells.append((r + k, c))
    cell_set = frozenset(cells)
    halo = set()
    for cell in cells:
        halo.update(neighbors8(cell))
    halo.difference_update(cell_set)
    return Placement(length, tuple(cells), orientation, cell_set, frozenset(halo))


def all_placements():
    out = {}
    for length in sorted(set(FLEET)):
        items = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE - length + 1):
                items.append(make_placement(r, c, length, "H"))
        for r in range(BOARD_SIZE - length + 1):
            for c in range(BOARD_SIZE):
                items.append(make_placement(r, c, length, "V"))
        out[length] = items
    return out


ALL_PLACEMENTS = all_placements()


def cluster_hits(hits):
    hits = set(hits)
    clusters = []

    while hits:
        start = next(iter(hits))
        stack = [start]
        comp = []
        hits.remove(start)

        while stack:
            cell = stack.pop()
            comp.append(cell)
            for nxt in neighbors4(cell):
                if nxt in hits:
                    hits.remove(nxt)
                    stack.append(nxt)

        comp.sort()
        clusters.append(tuple(comp))

    clusters.sort(key=len, reverse=True)
    return clusters


def cluster_orientation(cluster):
    rows = {r for r, _ in cluster}
    cols = {c for _, c in cluster}

    if len(cluster) <= 1:
        return None
    if len(rows) == 1:
        return "H"
    if len(cols) == 1:
        return "V"
    return "BAD"


def inferred_water(hits, sunk_ships, known_water):
    water = set(known_water)

    for ship in sunk_ships:
        for cell in ship:
            water.update(neighbors8(cell))
    for cell in hits:
        r, c = cell
        for dr in (-1, 1):
            for dc in (-1, 1):
                rr = r + dr
                cc = c + dc
                if inside(rr, cc):
                    water.add((rr, cc))

    for cluster in cluster_hits(hits):
        orient = cluster_orientation(cluster)
        if orient == "H":
            for r, c in cluster:
                if inside(r - 1, c):
                    water.add((r - 1, c))
                if inside(r + 1, c):
                    water.add((r + 1, c))
        elif orient == "V":
            for r, c in cluster:
                if inside(r, c - 1):
                    water.add((r, c - 1))
                if inside(r, c + 1):
                    water.add((r, c + 1))

    water.difference_update(hits)
    for ship in sunk_ships:
        water.difference_update(ship)
    return water


class HiddenBoard:
    def __init__(self, ships):
        self.ships = [tuple(ship) for ship in ships]
        self.ship_cells = set()
        self.cell_to_ship = {}
        self.hits = set()

        for idx, ship in enumerate(self.ships):
            for cell in ship:
                self.ship_cells.add(cell)
                self.cell_to_ship[cell] = idx

    def fire(self, cell):
        if cell in self.hits:
            raise ValueError("Cell was already fired at.")

        self.hits.add(cell)
        if cell not in self.ship_cells:
            return "miss", None

        ship_id = self.cell_to_ship[cell]
        ship = self.ships[ship_id]
        if all(part in self.hits for part in ship):
            return "sunk", ship

        return "hit", None

    def all_sunk(self):
        return self.ship_cells.issubset(self.hits)


def random_board(rng):
    ships = []
    forbidden = set()

    for length in sorted(FLEET, reverse=True):
        candidates = []
        for placement in ALL_PLACEMENTS[length]:
            if placement.cell_set & forbidden:
                continue
            candidates.append(placement)

        if not candidates:
            return random_board(rng)

        chosen = rng.choice(candidates)
        ships.append(chosen.cells)
        forbidden.update(chosen.cell_set)
        forbidden.update(chosen.halo)

    return HiddenBoard(ships)


class ObservationState:
    def __init__(self):
        self.misses = set()
        self.hits = set()
        self.sunk_ships = []
        self.remaining_lengths = list(FLEET)
        self.shots = []

    def fired_cells(self):
        cells = set(self.misses)
        cells.update(self.hits)
        for ship in self.sunk_ships:
            cells.update(ship)
        return cells

    def unknown_cells(self):
        blocked = self.fired_cells()
        out = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = (r, c)
                if cell not in blocked:
                    out.append(cell)
        return out

    def register_shot(self, cell, result, sunk_ship=None):
        self.shots.append((cell, result))

        if result == "miss":
            self.misses.add(cell)
            return

        if result == "hit":
            self.hits.add(cell)
            return

        ship = tuple(sorted(sunk_ship))
        for part in ship:
            self.hits.discard(part)
        self.sunk_ships.append(ship)
        self.remaining_lengths.remove(len(ship))

        halo = set()
        for part in ship:
            halo.update(neighbors8(part))
        halo.difference_update(ship)
        halo.difference_update(self.hits)
        for prev_ship in self.sunk_ships:
            halo.difference_update(prev_ship)
        self.misses.update(halo)

    def derived_water(self):
        return inferred_water(self.hits, self.sunk_ships, self.misses)


def valid_placement(placement, forbidden):
    return not (placement.cell_set & forbidden)


def placement_covers_cluster(placement, cluster):
    for cell in cluster:
        if cell not in placement.cell_set:
            return False
    return True


def cluster_candidates(cluster, length, forbidden, other_hits):
    orient = cluster_orientation(cluster)
    candidates = []

    for placement in ALL_PLACEMENTS[length]:
        if orient is not None and orient != "BAD" and len(cluster) > 1:
            if placement.orientation != orient:
                continue
        if not placement_covers_cluster(placement, cluster):
            continue
        if placement.cell_set & forbidden:
            continue
        if placement.cell_set & other_hits:
            continue
        if placement.halo & other_hits:
            continue
        candidates.append(placement)

    return candidates


def backtrack_clusters(clusters, lengths, forbidden, occupied, rng):
    if not clusters:
        return [], list(lengths), set(forbidden), set(occupied)

    cluster = clusters[0]
    rest = clusters[1:]
    other_hits = set()
    for item in rest:
        other_hits.update(item)

    tried_lengths = set()
    choices = []
    for length in lengths:
        if length in tried_lengths:
            continue
        if length < len(cluster):
            continue
        tried_lengths.add(length)
        choices.append(length)
    rng.shuffle(choices)

    for length in choices:
        candidates = cluster_candidates(cluster, length, forbidden, other_hits)
        rng.shuffle(candidates)
        for placement in candidates:
            next_lengths = list(lengths)
            next_lengths.remove(length)
            next_forbidden = set(forbidden)
            next_forbidden.update(placement.cell_set)
            next_forbidden.update(placement.halo)
            next_occupied = set(occupied)
            next_occupied.update(placement.cell_set)

            result = backtrack_clusters(rest, next_lengths, next_forbidden, next_occupied, rng)
            if result is not None:
                placed, left_lengths, fin_forbidden, fin_occupied = result
                return [placement] + placed, left_lengths, fin_forbidden, fin_occupied

    return None


def backtrack_fill(lengths, forbidden, rng, limit_per_level=40):
    if not lengths:
        return []

    length = lengths[0]
    candidates = []
    for placement in ALL_PLACEMENTS[length]:
        if placement.cell_set & forbidden:
            continue
        candidates.append(placement)

    rng.shuffle(candidates)
    if len(candidates) > limit_per_level:
        candidates = candidates[:limit_per_level]

    for placement in candidates:
        next_forbidden = set(forbidden)
        next_forbidden.update(placement.cell_set)
        next_forbidden.update(placement.halo)

        result = backtrack_fill(lengths[1:], next_forbidden, rng, limit_per_level)
        if result is not None:
            return [placement] + result

    return None


def sample_consistent_board(state, rng):
    forbidden = set(state.derived_water())
    occupied = set()

    for ship in state.sunk_ships:
        forbidden.update(ship)
        for cell in ship:
            forbidden.update(neighbors8(cell))

    clusters = cluster_hits(state.hits)
    for cluster in clusters:
        if cluster_orientation(cluster) == "BAD":
            return None

    cluster_result = backtrack_clusters(clusters, sorted(state.remaining_lengths, reverse=True), forbidden, occupied, rng)
    if cluster_result is None:
        return None

    cluster_placements, left_lengths, forbidden, occupied = cluster_result
    remaining = backtrack_fill(sorted(left_lengths, reverse=True), forbidden, rng)
    if remaining is None:
        return None

    placements = cluster_placements + remaining
    ship_cells = set()
    for placement in placements:
        ship_cells.update(placement.cell_set)

    if not state.hits.issubset(ship_cells):
        return None

    return ship_cells


def valid_ship_placements(state, length):
    forbidden = set(state.derived_water())
    hits = set(state.hits)

    for ship in state.sunk_ships:
        forbidden.update(ship)
        for cell in ship:
            forbidden.update(neighbors8(cell))

    placements = []

    if hits:
        seen = set()
        clusters = cluster_hits(hits)
        for cluster in clusters:
            other_hits = hits - set(cluster)
            for placement in cluster_candidates(cluster, length, forbidden, other_hits):
                key = placement.cells
                if key not in seen:
                    seen.add(key)
                    placements.append(placement)
    else:
        for placement in ALL_PLACEMENTS[length]:
            if not valid_placement(placement, forbidden):
                continue
            placements.append(placement)

    return placements


def posterior_heatmap(state, sample_count=None, rng=None):
    counts = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    unknown = set(state.unknown_cells())
    total_weight = 0

    for length, copies in Counter(state.remaining_lengths).items():
        placements = valid_ship_placements(state, length)
        total_weight += copies * len(placements)
        for placement in placements:
            for r, c in placement.cells:
                if (r, c) in unknown:
                    counts[r][c] += copies

    if total_weight == 0:
        return [[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)], 0

    heat = []
    for r in range(BOARD_SIZE):
        row = []
        for c in range(BOARD_SIZE):
            row.append(counts[r][c] / total_weight)
        heat.append(row)
    return heat, total_weight


class RandomStrategy:
    name = "random"

    def choose_shot(self, state, rng):
        return rng.choice(state.unknown_cells()), {"mode": "random"}


class HuntTargetStrategy:
    name = "hunt_target"

    def choose_shot(self, state, rng):
        unknown = set(state.unknown_cells())
        clusters = cluster_hits(state.hits)
        candidates = []

        for cluster in clusters:
            orient = cluster_orientation(cluster)

            if orient == "H":
                ordered = sorted(cluster, key=lambda cell: cell[1])
                r, c = ordered[0]
                left = (r, c - 1)
                r, c = ordered[-1]
                right = (r, c + 1)
                for cell in (left, right):
                    if cell in unknown:
                        candidates.append(cell)
            elif orient == "V":
                ordered = sorted(cluster, key=lambda cell: cell[0])
                r, c = ordered[0]
                up = (r - 1, c)
                r, c = ordered[-1]
                down = (r + 1, c)
                for cell in (up, down):
                    if cell in unknown:
                        candidates.append(cell)
            else:
                for hit in cluster:
                    for cell in neighbors4(hit):
                        if cell in unknown:
                            candidates.append(cell)

        if candidates:
            candidates = sorted(set(candidates), key=lambda cell: (-abs(cell[0] - 4.5) - abs(cell[1] - 4.5)))
            return candidates[0], {"mode": "target"}

        return rng.choice(state.unknown_cells()), {"mode": "hunt"}


class BayesianStrategy:
    name = "bayes"

    def __init__(self, sample_count=120):
        self.sample_count = sample_count

    def choose_shot(self, state, rng):
        heat, used = posterior_heatmap(state, self.sample_count, rng)
        best = None
        best_score = None
        candidates = state.unknown_cells()

        if not state.hits and state.remaining_lengths and min(state.remaining_lengths) > 1:
            parity_weight = [0.0, 0.0]
            for r, c in candidates:
                parity_weight[(r + c) % 2] += heat[r][c]
            wanted = 0 if parity_weight[0] >= parity_weight[1] else 1
            candidates = [cell for cell in candidates if (cell[0] + cell[1]) % 2 == wanted]

        for r, c in candidates:
            center_score = -abs(r - 4.5) - abs(c - 4.5)
            score = (heat[r][c], center_score)
            if best_score is None or score > best_score:
                best_score = score
                best = (r, c)

        return best, {"mode": "posterior", "counted_placements": used, "heat": heat}


def play_game(strategy, board, rng):
    state = ObservationState()
    shots = 0
    history = []

    while not board.all_sunk():
        shot, info = strategy.choose_shot(state, rng)
        result, sunk_ship = board.fire(shot)
        state.register_shot(shot, result, sunk_ship)
        shots += 1
        history.append((shot, result, info))

    return {
        "shots": shots,
        "history": history,
        "state": state,
    }


def summarize(values):
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n

    def q(p):
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return values[idx]

    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": values[0],
        "median": q(0.5),
        "p10": q(0.1),
        "p90": q(0.9),
        "max": values[-1],
    }


def render_heatmap_svg(heat, path, title, best_cell=None, shots=None, true_board=None):
    cell = 44
    margin = 48
    width = margin * 2 + BOARD_SIZE * cell
    top = 70
    height = top + margin + BOARD_SIZE * cell

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
        f'<text x="{margin}" y="34" font-size="24" font-family="Arial" fill="#222">{title}</text>',
    ]

    for i in range(BOARD_SIZE):
        lines.append(
            f'<text x="{margin + i * cell + 16}" y="{top - 14}" font-size="14" font-family="Arial" fill="#333">{i + 1}</text>'
        )
        lines.append(
            f'<text x="{margin - 24}" y="{top + i * cell + 27}" font-size="14" font-family="Arial" fill="#333">{chr(ord("A") + i)}</text>'
        )

    max_p = max(max(row) for row in heat) if heat else 1.0
    if max_p == 0:
        max_p = 1.0

    shot_cells = {}
    if shots is not None:
        for cell_pos, result in shots:
            shot_cells[cell_pos] = result

    ship_cells = set()
    if true_board is not None:
        ship_cells = set(true_board.ship_cells)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            x = margin + c * cell
            y = top + r * cell
            p = heat[r][c] / max_p
            red = 255
            green = int(245 - 120 * p)
            blue = int(240 - 180 * p)
            fill = f"rgb({red},{green},{blue})"

            if true_board is not None and (r, c) in ship_cells:
                fill = "#dcefdc"

            if (r, c) in shot_cells:
                if shot_cells[(r, c)] == "miss":
                    fill = "#d8e7f7"
                else:
                    fill = "#f7d8d8"

            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#777"/>')
            lines.append(
                f'<text x="{x + 7}" y="{y + 26}" font-size="12" font-family="Arial" fill="#333">{heat[r][c]:.2f}</text>'
            )

            if (r, c) in shot_cells:
                mark = "•" if shot_cells[(r, c)] == "miss" else "×"
                lines.append(
                    f'<text x="{x + 27}" y="{y + 27}" font-size="18" font-family="Arial" fill="#111">{mark}</text>'
                )

            if best_cell == (r, c):
                lines.append(
                    f'<rect x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" fill="none" stroke="#111" stroke-width="3"/>'
                )

    lines.append("</svg>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_histogram_svg(series, path, title):
    all_values = []
    for _, values in series:
        all_values.extend(values)

    lo = min(all_values)
    hi = max(all_values)
    bin_size = 4
    bins = list(range((lo // bin_size) * bin_size, ((hi // bin_size) + 2) * bin_size, bin_size))
    counts = []

    for name, values in series:
        hist = []
        for left in bins[:-1]:
            right = left + bin_size
            hist.append(sum(1 for x in values if left <= x < right))
        counts.append((name, hist))

    max_count = max(max(hist) for _, hist in counts)
    width = 920
    height = 520
    left_margin = 70
    bottom = 440
    plot_w = 780
    plot_h = 320
    bar_group = plot_w / len(counts[0][1])
    bar_w = bar_group / (len(series) + 1)
    colors = ["#d15b47", "#4b83c3", "#6d9f4b"]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
        f'<text x="{left_margin}" y="34" font-size="24" font-family="Arial" fill="#222">{title}</text>',
    ]

    for i in range(6):
        y = bottom - plot_h * i / 5
        value = round(max_count * i / 5)
        lines.append(f'<line x1="{left_margin}" y1="{y}" x2="{left_margin + plot_w}" y2="{y}" stroke="#ddd"/>')
        lines.append(f'<text x="20" y="{y + 5}" font-size="12" font-family="Arial" fill="#444">{value}</text>')

    for idx, (name, hist) in enumerate(counts):
        for b, count in enumerate(hist):
            x = left_margin + b * bar_group + idx * bar_w
            h = 0 if max_count == 0 else plot_h * count / max_count
            y = bottom - h
            lines.append(
                f'<rect x="{x}" y="{y}" width="{bar_w - 4}" height="{h}" fill="{colors[idx % len(colors)]}" opacity="0.85"/>'
            )

    for b, left in enumerate(bins[:-1]):
        x = left_margin + b * bar_group + 4
        label = f"{left}-{left + bin_size - 1}"
        lines.append(f'<text x="{x}" y="{bottom + 20}" font-size="11" font-family="Arial" fill="#333">{label}</text>')

    for idx, (name, _) in enumerate(counts):
        x = left_margin + idx * 180
        y = 480
        lines.append(f'<rect x="{x}" y="{y - 12}" width="18" height="18" fill="{colors[idx % len(colors)]}"/>')
        lines.append(f'<text x="{x + 28}" y="{y + 2}" font-size="14" font-family="Arial" fill="#222">{name}</text>')

    lines.append("</svg>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
