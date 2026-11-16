"""Freeze train-only DFDC original/fake pairs and mixed-source replay."""
import random
from collections import Counter, defaultdict

from training.artifact_spatial.sampling import MatchedRotation
from training.spatial.common import fingerprint
from .flow import motion_bucket


def build_plans(records, tried, args):
    if args.videos_per_epoch % 4 or args.max_fake_source_fraction < .5:
        raise ValueError('Paired batches need a multiple of four videos and a DFDC cap of at least 50%')
    rng = random.Random(args.seed)
    pools = defaultdict(lambda: ([], []))
    replay = []
    for row in records:
        if row['source'] == 'DFDC':
            if row['label'] == 0 or row['id'] not in tried:
                pools[row['group'], motion_bucket(row)][row['label']].append(row)
        else:
            replay.append(row)
    groups = defaultdict(list)
    for (group, cell), (real, fake) in sorted(pools.items()):
        if real and fake:
            groups[group].append(tuple(rng.choice(sorted(side, key=lambda r: r['id'])) for side in (real, fake)))
    pairs = [rng.choice(options) for options in groups.values()]
    rng.shuffle(pairs)
    per_epoch = args.videos_per_epoch // 4
    if len(pairs) < per_epoch * args.epochs:
        raise ValueError(f'Only {len(pairs)} distinct DFDC pairs with previously unused fakes')
    rotation = MatchedRotation(replay, seed=args.seed, previously_seen=tried, bucket=motion_bucket,
                               prefer_unseen=True,
                               max_source_fraction_by_class={1: args.max_fake_source_fraction})
    plans, seen = [], set(tried)
    for index in range(args.epochs):
        dfdc = pairs[index*per_epoch:(index+1)*per_epoch]
        mixed = rotation.take(per_epoch*2)
        selection = [row for pair in dfdc for row in pair] + mixed
        rng.shuffle(selection)
        novel = [row for row in selection if row['id'] not in seen]
        seen.update(row['id'] for row in selection)
        plans.append(dict(epoch=index+1, ids=[row['id'] for row in selection],
                          pairs=[[row['id'] for row in pair] for pair in dfdc],
                          sources=dict(Counter(row['source'] for row in selection)),
                          new_by_label=dict(Counter(str(row['label']) for row in novel))))
    payload = dict(plans=plans, paired_groups=len(pairs), new_total=len(seen)-len(tried),
                   tried_before=len(tried))
    return dict(payload, fingerprint=fingerprint(payload))


def verify_plans(payload, fingerprint_expected, eligible, epochs, count):
    body = {k: v for k, v in payload.items() if k != 'fingerprint'}
    if (payload['fingerprint'] != fingerprint_expected or fingerprint(body) != fingerprint_expected or
            len(payload['plans']) != epochs):
        raise ValueError('Frozen paired-batch plan changed')
    for number, plan in enumerate(payload['plans'], 1):
        ids = set(plan['ids'])
        if plan['epoch'] != number or len(ids) != count or len(plan['ids']) != count or not ids <= eligible:
            raise ValueError(f'Frozen paired-batch epoch {number} is incompatible with the train cohort')
