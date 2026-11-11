"""Versioned, deterministic interpretation of spatial and motion scores."""

import math

FRAME_STEP = 2.
CLIP_STEP = 1.5
THRESHOLD = .65
REVIEW_SCORE = .35
DISAGREEMENT = .35
SPATIAL_WEIGHT = .7
TEMPORAL_WEIGHT = .3
MERGE_GAP = .05
FRAME_NEIGHBOR_MAX = 2.25
MOTION_NEIGHBOR_MIN = .75
MOTION_NEIGHBOR_MAX = CLIP_STEP + .05
CROSS_OVERLAP_MIN = .5
POLICY = 'trace-evidence-v3'


def merge_spans(spans, duration):
    merged = []
    for start, end, reason in sorted(spans):
        start, end = max(0., start), min(duration, end)
        if start >= end:
            continue
        if merged and start <= merged[-1]['end'] + MERGE_GAP:
            merged[-1]['end'] = max(end, merged[-1]['end'])
            if reason not in merged[-1]['evidence']:
                merged[-1]['evidence'].append(reason)
        else:
            merged.append(dict(start=start, end=end, evidence=[reason]))
    return merged


def segments(frames, clips, duration, threshold=THRESHOLD):
    """Legacy OR rule, retained as a benchmark control."""
    spans = [(f['time'], f['time'] + FRAME_STEP, 'spatial') for f in frames
             if f.get('spatial', 0) >= threshold and 0 <= f['time'] < duration]
    spans += [(c['start'], c['end'], 'temporal') for c in clips if c.get('temporal', 0) >= threshold]
    return [dict(start=s['start'], end=s['end']) for s in merge_spans(spans, duration)]


def supported_spans(frames, clips, duration):
    repeated, shared = [], []
    for a, b in zip(frames, frames[1:]):
        if (min(a['spatial'], b['spatial']) >= THRESHOLD and
                0 < b['time'] - a['time'] <= FRAME_NEIGHBOR_MAX):
            repeated.append((a['time'], b['time'] + FRAME_STEP, 'spatial_persistence'))
    for a, b in zip(clips, clips[1:]):
        if (min(a['temporal'], b['temporal']) >= THRESHOLD and
                MOTION_NEIGHBOR_MIN <= b['start'] - a['start'] <= MOTION_NEIGHBOR_MAX):
            repeated.append((a['start'], b['end'], 'motion_persistence'))
    for frame in frames:
        t = frame['time']
        if frame['spatial'] < THRESHOLD:
            continue
        for clip in clips:
            end = min(t + FRAME_STEP, clip['end'], duration)
            if clip['temporal'] >= THRESHOLD and clip['start'] <= t and end - t >= CROSS_OVERLAP_MIN:
                shared.append((t, end, 'cross_branch'))
    confirmed = merge_spans(shared, duration)
    review = [s for s in merge_spans(repeated, duration)
              if not any(s['start'] < c['end'] and c['start'] < s['end'] for c in confirmed)]
    return confirmed, review


def assess(frames, clips, duration, expected_clips):
    if not math.isfinite(duration) or duration <= 0 or not frames or expected_clips < len(clips):
        raise ValueError('Assessment requires spatial evidence, duration, and consistent coverage')
    for frame in frames:
        if not 0 <= frame['time'] < duration or not 0 <= frame['spatial'] <= 1:
            raise ValueError('Invalid spatial timestamp or score')
    for clip in clips:
        if not 0 <= clip['start'] < clip['end'] <= duration + 1e-6 or not 0 <= clip['temporal'] <= 1:
            raise ValueError('Invalid temporal interval or score')
    frames = sorted(frames, key=lambda frame: frame['time'])
    clips = sorted(clips, key=lambda clip: clip['start'])
    spatial = sum(f['spatial'] for f in frames) / len(frames)
    temporal = sum(c['temporal'] for c in clips) / len(clips) if clips else None
    combined = SPATIAL_WEIGHT * spatial + TEMPORAL_WEIGHT * temporal if temporal is not None else spatial
    disagreement = temporal is not None and abs(spatial - temporal) >= DISAGREEMENT
    motion_status = ('complete' if len(clips) == expected_clips and clips else
                     'partial' if clips else 'failed' if expected_clips else 'unsupported')
    raw = segments(frames, clips, duration)
    regions, review = supported_spans(frames, clips, duration)
    reasons = []
    if regions:
        verdict = 'suspected'
        reasons.append('cross_branch_intervals')
    else:
        if review:
            reasons.append('single_branch_persistence')
        elif raw:
            reasons.append('isolated_high_scores')
        if combined >= REVIEW_SCORE:
            reasons.append('elevated_mean_score')
        if disagreement:
            reasons.append('branch_disagreement')
        if motion_status != 'complete':
            reasons.append('motion_not_complete')
        verdict = 'uncertain' if reasons else 'no_signals'
        if not reasons:
            reasons.append('low_scores_with_complete_motion')
    return dict(policy=POLICY, verdict=verdict, reasons=reasons, spatial=spatial, temporal=temporal,
                combined=combined, threshold=THRESHOLD, disagreement=disagreement, segments=regions,
                review_segments=review, candidate_segments=raw,
                suspicious_seconds=sum(s['end'] - s['start'] for s in regions),
                spatial_samples=len(frames), temporal_samples=len(clips),
                coverage=dict(motion=motion_status, expected_clips=expected_clips, scored_clips=len(clips)),
                parameters=dict(sample_threshold=THRESHOLD, review_score=REVIEW_SCORE,
                                disagreement_threshold=DISAGREEMENT, spatial_weight=SPATIAL_WEIGHT,
                                temporal_weight=TEMPORAL_WEIGHT, frame_step=FRAME_STEP, clip_step=CLIP_STEP,
                                frame_neighbor_max=FRAME_NEIGHBOR_MAX, motion_neighbor_min=MOTION_NEIGHBOR_MIN,
                                motion_neighbor_max=MOTION_NEIGHBOR_MAX, cross_overlap_min=CROSS_OVERLAP_MIN,
                                merge_gap=MERGE_GAP),
                caveat='Scores are not calibrated probabilities. Intervals are evidence for review, not proof or masks. '
                       'Overlapping clips can share frames. The rule is heuristic and requires independent validation.')
