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
                 bucket=spatial_bucket, max_source_fraction_by_class=None):
        self.records = {r["id"]: r for r in records}
        self.pools = defaultdict(list)
        for record in records:
            key = (bucket(record), record["label"], record["source"])
            self.pools[key].append(record["id"])
        self.cells = sorted({key[0] for key in self.pools
                             if any(k[0] == key[0] and k[1] != key[1] for k in self.pools)})
        self.seed, self.inherited, self.ucf_fraction = seed, set(previously_seen or []), ucf_fraction
        self.max_source_fraction_by_class = dict(max_source_fraction_by_class or {})
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
        counts, queues = Counter(), {}
        cap = math.floor(target * self.ucf_fraction)
        positions = self.state["positions"]

        def candidate(key):
            fraction = self.max_source_fraction_by_class.get(key[1], 1.)
            if counts[(key[1], key[2])] >= math.floor(target*fraction):
                return None
            if key[2] == "UCF-101" and key[1] == 0 and counts[(0, key[2])] >= cap:
                return None
            pool, name = self.pools[key], "|".join(map(str, key))
            if all(item in used for item in pool):
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
                    return item
                positions[name] = position + 1
            raise RuntimeError("Sampler could not locate an unused item")

        while len(selected) < count:
            paired = False
            for _ in self.cells:
                cell = self.cells[self.state["turn"] % len(self.cells)]
                self.state["turn"] += 1
                pair = []
                for label in (0, 1):
                    keys = sorted((key for key in self.pools if key[:2] == (cell, label)),
                                  key=lambda k: (counts[(label, k[2])],
                                                 fingerprint([self.seed, self.state["turn"], k])))
                    options = [(key, candidate(key)) for key in keys]
                    available = next(((key, item) for key, item in options if item is not None), None)
                    if available is None:
                        break
                    pair.append(available)
                if len(pair) != 2:
                    continue
                for key, item in pair:
                    positions["|".join(map(str, key))] = positions.get("|".join(map(str, key)), 0) + 1
                    selected.append(self.records[item])
                    counts[(key[1], key[2])] += 1
                    used.add(item)
                paired = True
                break
            if not paired:
                raise ValueError(f"Matched pool/caps supply only {len(selected)}/{count} unique records. "
                                 "Use a smaller videos-per-epoch in a NEW run; no silent imbalance.")
        random.Random(f"{self.seed}:{self.state}").shuffle(selected)
        return selected
