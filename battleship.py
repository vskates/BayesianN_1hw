from collections import Counter


BOARD_SIZE = 10
FLEET = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]


def inside(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def neigh4(cell):
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if inside(rr, cc):
            out.append((rr, cc))
    return out


def neigh8(cell):
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


def ship_dir(cells):
    if len(cells) <= 1:
        return None
    if cells[0][0] == cells[1][0]:
        return "H"
    return "V"


def ship_halo(cells):
    out = set()
    for cell in cells:
        out |= neigh8(cell)
    return out - set(cells)


def make_all_positions():
    pos = {}
    for n in sorted(set(FLEET)):
        cur = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE - n + 1):
                cells = tuple((r, c + k) for k in range(n))
                cur.append((cells, ship_halo(cells), "H"))
        for r in range(BOARD_SIZE - n + 1):
            for c in range(BOARD_SIZE):
                cells = tuple((r + k, c) for k in range(n))
                cur.append((cells, ship_halo(cells), "V"))
        pos[n] = cur
    return pos


ALL_POS = make_all_positions()


def clusters(hits):
    hits = set(hits)
    out = []
    while hits:
        start = hits.pop()
        stack = [start]
        cur = [start]
        while stack:
            v = stack.pop()
            for u in neigh4(v):
                if u in hits:
                    hits.remove(u)
                    stack.append(u)
                    cur.append(u)
        cur.sort()
        out.append(tuple(cur))
    out.sort(key=len, reverse=True)
    return out


def inferred_water(hits, sunk, misses):
    water = set(misses)

    for ship in sunk:
        water |= ship_halo(ship)

    for r, c in hits:
        for dr in (-1, 1):
            for dc in (-1, 1):
                if inside(r + dr, c + dc):
                    water.add((r + dr, c + dc))

    for cl in clusters(hits):
        d = ship_dir(cl)
        if d == "H":
            for r, c in cl:
                if inside(r - 1, c):
                    water.add((r - 1, c))
                if inside(r + 1, c):
                    water.add((r + 1, c))
        if d == "V":
            for r, c in cl:
                if inside(r, c - 1):
                    water.add((r, c - 1))
                if inside(r, c + 1):
                    water.add((r, c + 1))

    for ship in sunk:
        water -= set(ship)
    return water - set(hits)


class HiddenBoard:
    def __init__(self, ships):
        self.ships = [tuple(ship) for ship in ships]
        self.ship_cells = set()
        self.which_ship = {}
        self.opened = set()
        for i, ship in enumerate(self.ships):
            for cell in ship:
                self.ship_cells.add(cell)
                self.which_ship[cell] = i

    def fire(self, cell):
        self.opened.add(cell)
        if cell not in self.ship_cells:
            return "miss", None
        ship = self.ships[self.which_ship[cell]]
        if all(x in self.opened for x in ship):
            return "sunk", ship
        return "hit", None

    def all_sunk(self):
        return self.ship_cells <= self.opened


def random_board(rng):
    while True:
        busy = set()
        ships = []
        ok = True
        for n in sorted(FLEET, reverse=True):
            cand = []
            for cells, halo, d in ALL_POS[n]:
                if set(cells) & busy:
                    continue
                cand.append((cells, halo))
            if not cand:
                ok = False
                break
            cells, halo = rng.choice(cand)
            ships.append(cells)
            busy |= set(cells)
            busy |= halo
        if ok:
            return HiddenBoard(ships)


class ObservationState:
    def __init__(self):
        self.misses = set()
        self.hits = set()
        self.sunk_ships = []
        self.remaining_lengths = list(FLEET)
        self.shots = []

    def fired_cells(self):
        out = set(self.misses) | set(self.hits)
        for ship in self.sunk_ships:
            out |= set(ship)
        return out

    def unknown_cells(self):
        bad = self.fired_cells()
        out = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r, c) not in bad:
                    out.append((r, c))
        return out

    def derived_water(self):
        return inferred_water(self.hits, self.sunk_ships, self.misses)

    def register_shot(self, cell, result, sunk_ship=None):
        self.shots.append((cell, result))
        if result == "miss":
            self.misses.add(cell)
        elif result == "hit":
            self.hits.add(cell)
        else:
            ship = tuple(sorted(sunk_ship))
            self.sunk_ships.append(ship)
            self.remaining_lengths.remove(len(ship))
            self.hits -= set(ship)
            self.misses |= ship_halo(ship)
            for old in self.sunk_ships:
                self.misses -= set(old)
            self.misses -= self.hits


def possible_places(state, n):
    water = state.derived_water()
    hits = set(state.hits)
    out = []
    seen = set()
    cls = clusters(hits)

    for cells, halo, d in ALL_POS[n]:
        cells_set = set(cells)
        if cells_set & water:
            continue

        if not hits:
            out.append((cells, halo, d))
            continue

        for cl in cls:
            if len(cl) > n:
                continue
            if len(cl) > 1 and ship_dir(cl) != d:
                continue
            if not set(cl) <= cells_set:
                continue
            other_hits = hits - set(cl)
            if cells_set & other_hits:
                continue
            if halo & other_hits:
                continue
            if cells not in seen:
                out.append((cells, halo, d))
                seen.add(cells)
            break

    return out


def posterior_heatmap(state, sample_count=None, rng=None):
    heat = [[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    unknown = set(state.unknown_cells())
    total = 0

    for n, k in Counter(state.remaining_lengths).items():
        pos = possible_places(state, n)
        total += k * len(pos)
        for cells, halo, d in pos:
            for r, c in cells:
                if (r, c) in unknown:
                    heat[r][c] += k

    if total == 0:
        return heat, 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            heat[r][c] /= total
    return heat, total


class RandomStrategy:
    name = "random"

    def choose_shot(self, state, rng):
        return rng.choice(state.unknown_cells()), {"mode": "random"}

class BayesianStrategy:
    name = "bayes"

    def __init__(self, sample_count=0):
        self.sample_count = sample_count

    def choose_shot(self, state, rng):
        heat, total = posterior_heatmap(state)
        cand = state.unknown_cells()

        if not state.hits and state.remaining_lengths and min(state.remaining_lengths) > 1:
            p0 = sum(heat[r][c] for r, c in cand if (r + c) % 2 == 0)
            p1 = sum(heat[r][c] for r, c in cand if (r + c) % 2 == 1)
            need = 0 if p0 >= p1 else 1
            cand = [x for x in cand if (x[0] + x[1]) % 2 == need]

        best = max(cand, key=lambda x: (heat[x[0]][x[1]], -abs(x[0] - 4.5) - abs(x[1] - 4.5)))
        return best, {"mode": "posterior", "counted_placements": total, "heat": heat}


def play_game(strategy, board, rng):
    state = ObservationState()
    shots = 0
    history = []

    while not board.all_sunk():
        shot, info = strategy.choose_shot(state, rng)
        result, sunk_ship = board.fire(shot)
        state.register_shot(shot, result, sunk_ship)
        history.append((shot, result, info))
        shots += 1

    return {"shots": shots, "history": history, "state": state}


def summarize(values):
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    std = (sum((x - mean) ** 2 for x in values) / n) ** 0.5

    def q(p):
        return values[round((n - 1) * p)]

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": values[0],
        "median": q(0.5),
        "p10": q(0.1),
        "p90": q(0.9),
        "max": values[-1],
    }


def render_heatmap_svg(heat, path, title, best_cell=None, shots=None, true_board=None):
    cell = 44
    left = 48
    top = 70
    w = left * 2 + BOARD_SIZE * cell
    h = top + 60 + BOARD_SIZE * cell
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
        f'<text x="{left}" y="34" font-size="24" font-family="Arial" fill="#222">{title}</text>',
    ]

    shot_map = {}
    if shots:
        for cell_pos, result in shots:
            shot_map[cell_pos] = result
    ship_cells = set(true_board.ship_cells) if true_board is not None else set()
    mx = max(max(row) for row in heat) if heat else 1
    if mx == 0:
        mx = 1

    for i in range(BOARD_SIZE):
        lines.append(f'<text x="{left + i * cell + 16}" y="{top - 14}" font-size="14">{i + 1}</text>')
        lines.append(f'<text x="{left - 24}" y="{top + i * cell + 27}" font-size="14">{chr(ord("A") + i)}</text>')

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            x = left + c * cell
            y = top + r * cell
            p = heat[r][c] / mx
            fill = f"rgb(255,{int(245 - 120 * p)},{int(240 - 180 * p)})"
            if (r, c) in ship_cells:
                fill = "#dcefdc"
            if (r, c) in shot_map:
                fill = "#d8e7f7" if shot_map[(r, c)] == "miss" else "#f7d8d8"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#777"/>')
            lines.append(f'<text x="{x + 7}" y="{y + 26}" font-size="12">{heat[r][c]:.2f}</text>')
            if best_cell == (r, c):
                lines.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" fill="none" stroke="#111" stroke-width="3"/>')

    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_histogram_svg(series, path, title):
    nums = [x for _, values in series for x in values]
    lo = min(nums)
    hi = max(nums)
    step = 4
    bins = list(range((lo // step) * step, ((hi // step) + 2) * step, step))
    hists = []
    for name, values in series:
        cur = []
        for left in bins[:-1]:
            cur.append(sum(1 for x in values if left <= x < left + step))
        hists.append((name, cur))

    mx = max(max(cur) for _, cur in hists)
    w = 920
    h = 520
    left = 70
    bottom = 440
    pw = 780
    ph = 320
    bw = pw / len(hists[0][1]) / (len(series) + 1)
    group = pw / len(hists[0][1])
    colors = ["#d15b47", "#4b83c3", "#6d9f4b"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
        f'<text x="{left}" y="34" font-size="24">{title}</text>',
    ]

    for i in range(6):
        y = bottom - ph * i / 5
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left + pw}" y2="{y}" stroke="#ddd"/>')

    for k, (name, cur) in enumerate(hists):
        for i, cnt in enumerate(cur):
            x = left + i * group + k * bw
            hh = 0 if mx == 0 else ph * cnt / mx
            y = bottom - hh
            lines.append(f'<rect x="{x}" y="{y}" width="{bw - 4}" height="{hh}" fill="{colors[k]}" opacity="0.85"/>')

    for i, left_bin in enumerate(bins[:-1]):
        x = left + i * group + 4
        lines.append(f'<text x="{x}" y="{bottom + 20}" font-size="11">{left_bin}-{left_bin + step - 1}</text>')

    for k, (name, cur) in enumerate(hists):
        x = left + k * 180
        y = 480
        lines.append(f'<rect x="{x}" y="{y - 12}" width="18" height="18" fill="{colors[k]}"/>')
        lines.append(f'<text x="{x + 28}" y="{y + 2}" font-size="14">{name}</text>')

    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
