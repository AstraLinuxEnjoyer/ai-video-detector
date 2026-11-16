"""Train-only matching on coarse spatial properties, with resumable source rotation."""
import copy
import math
import random
from collections import Counter, defaultdict

from training.spatial.common import fingerprint


def spatial_bucket(record):
    meta = record.get("metadata") or {}
    width = meta.get("width", record.get("sequence_width", 0))
    height = meta.get("height", record.get("sequence_height", 0))
    if not width or not height:
        raise ValueError(f"Missing audited dimensions: {record['id']}")
    short = min(width, height)
    resolution = "low" if short < 360 else "medium" if short < 720 else "high"
    aspect = "portrait" if width/height < .8 else "landscape" if width/height > 1.25 else "square"
    return resolution + "/" + aspect


def distribution(records, bucket=spatial_bucket):
    cells = defaultdict(Counter)
    for record in records:
        cells[bucket(record)][str(record["label"])] += 1
    return {key: dict(value) for key, value in sorted(cells.items())}


class MatchedRotation:
    """Equal real/fake counts per cell; fair sources, no within-epoch duplicates.

    Bins do not equate visual quality or codec history. Unmatched records stay in
    the frozen manifest and are explicitly reported, never moved into another split.
    """
    def __init__(self, records, seed=42, state=None, previously_seen=None, ucf_fraction=.2,
                 bucket=spatial_bucket, max_source_fraction_by_class=None, prefer_unseen=False):
        self.records = {r["id"]: r for r in records}
        self.pools = defaultdict(list)
        for record in records:
            key = (bucket(record), record["label"], record["source"])
            self.pools[key].append(record["id"])
        self.cells = sorted({key[0] for key in self.pools
                             if any(k[0] == key[0] and k[1] != key[1] for k in self.pools)})
        self.seed, self.inherited, self.ucf_fraction = seed, set(previously_seen or []), ucf_fraction
        self.max_source_fraction_by_class = dict(max_source_fraction_by_class or {})
        self.prefer_unseen = prefer_unseen
        self.keys_by_cell_label = defaultdict(list)
        for key in sorted(self.pools):
            self.keys_by_cell_label[key[:2]].append(key)
        self.state = copy.deepcopy(state) if state else dict(positions={}, turn=0)
        self.audit = dict(pool=distribution(records, bucket), matched_cells=self.cells,
                          unmatched_ids=[r["id"] for r in records if bucket(r) not in self.cells],
                          unmatched_by_source=dict(Counter(r["source"] for r in records if bucket(r) not in self.cells)),
                          caveat="Coarse resolution/aspect matching only; not semantic or codec debiasing")
        if not self.cells:
            raise ValueError("No resolution/aspect cells contain both classes")

    def take(self, count):
        if count < 2 or count % 2:
            raise ValueError("Matched sampling needs a positive even video count")
        target, used, selected = count//2, set(), []
        counts, used_per_pool, queues = Counter(), Counter(), {}
        cap = math.floor(target * self.ucf_fraction)
        positions = self.state["positions"]

        def candidate(key):
            fraction = self.max_source_fraction_by_class.get(key[1], 1.)
            if counts[(key[1], key[2])] >= math.floor(target*fraction):
                return None
            if key[2] == "UCF-101" and key[1] == 0 and counts[(0, key[2])] >= cap:
                return None
            pool, name = self.pools[key], "|".join(map(str, key))
            if used_per_pool[key] == len(pool):
                return None
            for _ in range(2 * len(pool)):
                position = positions.get(name, 0)
                cycle, offset = divmod(position, len(pool))
                if (key, cycle) not in queues:
                    queue = sorted(pool)
                    random.Random(f"{self.seed}:{name}:{cycle}").shuffle(queue)
                    if cycle == 0:
                        queue.sort(key=lambda item: item in self.inherited)
                    queues[key, cycle] = queue
                item = queues[key, cycle][offset]
                if item not in used:
                    return item, cycle == 0 and item not in self.inherited
                positions[name] = position + 1
            raise RuntimeError("Sampler could not locate an unused item")

        while len(selected) < count:
            start = self.state["turn"]
            best, best_novel, next_turn = None, -1, start
            for i in range(len(self.cells)):
                turn = start + i + 1
                cell = self.cells[(turn-1) % len(self.cells)]
                pair = []
                for label in (0, 1):
                    keys = sorted(self.keys_by_cell_label[cell, label],
                                  key=lambda k: (counts[(label, k[2])],
                                                 fingerprint([self.seed, turn, k])))
                    choice = None
                    if not self.prefer_unseen:
                        options = [(key, candidate(key)) for key in keys]
                        choice = next(((key, *hit) for key, hit in options if hit is not None), None)
                    else:
                        for key in keys:
                            hit = candidate(key)
                            if hit is not None and (choice is None or hit[1]):
                                choice = (key, *hit)
                                if hit[1]:
                                    break
                    if choice is None:
                        break
                    pair.append(choice)
                if len(pair) != 2:
                    continue
                novel = sum(item[2] for item in pair)
                if novel > best_novel:
                    best, best_novel, next_turn = pair, novel, turn
                if not self.prefer_unseen or novel == 2:
                    break
            if best is None:
                raise ValueError(f"Matched pool/caps supply only {len(selected)}/{count} unique records. "
                                 "Use a smaller videos-per-epoch in a NEW run; no silent imbalance.")
            self.state["turn"] = next_turn
            for key, item, _ in best:
                name = "|".join(map(str, key))
                positions[name] = positions.get(name, 0) + 1
                selected.append(self.records[item])
                counts[(key[1], key[2])] += 1
                used_per_pool[key] += 1
                used.add(item)
        random.Random(f"{self.seed}:{self.state}").shuffle(selected)
        return selected
