# Motion-only temporal training

This is a separate experiment, not a conversion of the completed RGB checkpoint.
The first model and the existing `r2plus1d_v1` weights remain untouched.

## Launch

From PowerShell in `C:\Users\Ivan\Desktop\ai-video-detector`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train-temporal.ps1 -MotionTrain
```

The wrapper selects the working Python runtime and invokes
`training.temporal_v2.run --motion`. Restart the identical command to resume.
No `--fresh` is needed. Do not point the motion run at the RGB run directory.

The console prints `DASHBOARD http://127.0.0.1:8766` (or the next available port).
After training exits, open the persisted dashboard with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train-temporal.ps1 -DashboardOnly --run-dir cache\temporal_v2\runs\motion_flow_v1
```

For independent, correctly labeled videos in `test_videos\real` and
`test_videos\fake`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train-temporal.ps1 -MotionTrain --evaluate-only
```

Evaluation requires `best.pt`, created after at least one completed epoch. Training
files are rejected as external tests. Low-FPS/short videos receive no prediction;
the coverage report includes them in the requested denominator.

## Actual input

1. Reuse the frozen spatial partitions: content hashes and known origin groups do
   not migrate between train/validation/test. No new datasets are downloaded.
2. Read existing train/validation video headers. Cache progress in
   `motion_headers.jsonl`; a restart does not re-open all headers.
3. Sample nine RGB frames at four frames per second from each clip. Require native
   average FPS >= 4 and duration >= 2.25 seconds. No interpolation or deliberate
   upsampling of low-FPS sources. Average FPS does not prove constant timing;
   pre-existing frame holds/static frames are logged, not relabeled as fake.
4. Compute eight dense flow fields using OpenCV Farneback on grayscale images.
   Subtract the per-pair spatial median vector: this removes approximate global
   translation, NOT camera rotation, zoom, parallax or full camera motion.
5. Convert displacements to normalized frame-width units per second. Normalize
   both signed velocity channels with the same fixed scale 0.5, then clamp to
   [-1, 1]. There is no per-video maximum normalization.
6. R(2+1)D-18 sees ONLY vx/vy, shaped `2 x 8 x 112 x 112`. RGB is used by the flow
   estimator and the reference preview, never concatenated into the classifier.
   Its two-channel stem is initialized from averaged Kinetics weights; other
   pretrained blocks transfer. RGB BatchNorm running statistics are reset and
   learned on motion. No previous fake-detector checkpoint initializes this run.
7. Crop/resize the whole flow sequence consistently. Horizontal mirroring negates
   vx; crop scaling adjusts velocity units. No RGB brightness/color/JPEG transform
   is incorrectly applied to vector channels.

Flow computation and video decoding run on CPU before GPU optimization. The
classifier is trained on the GPU. A CPU-heavy preparation phase is expected.
The bounded cache holds decoded frames and float16 flow; originals are unchanged.
Train caches are removed between epochs. Default budget is 12 GiB with 10 GiB
of free-space reserve, excluding model checkpoints and originals.

## Sampling and counts

Each epoch selects 500 videos: 250 real and 250 fake, two clips each. Source
rotation is resumable and prevents within-epoch duplicate IDs. Counts are balanced
inside coarse resolution/aspect/native-FPS cells. A single fake source cannot
exceed 60% of that epoch's fake videos. Larger 1,000-video plans could not meet
this cap on the current matched pool; the count is deliberately reduced rather
than filling the plan with Pika or silently losing balance.
FPS bands are 4-8, 8-16, 16-24,
24-32, 32+. Matching is not exact codec, content or visual-quality debiasing.
Validation selects the same 800 videos, 400 per class, with three clips each,
using class/source rotation within the eligible pool. Unlike training, its
selected FPS/spatial-cell counts are not required to match between classes:
strict per-cell matching made the 800-video validation 387/400 Pika on the fake
side. Source coverage and per-native-FPS metrics expose that tradeoff instead.
The validation IDs are fixed by the frozen pool and seed, not reselected to improve
metrics. The completed RGB validation used a different cohort; compare common
held-out IDs before claiming the motion model is better.

The 2026-10-08 header audit of the current snapshot found:

- Train: 17,150 eligible videos, 4,725 real and 12,425 fake.
- Validation pool: 2,118 videos, 618 real and 1,500 fake.
- Train input-contract exclusions: 33,183; unmatched FPS/spatial cells: 11,308.
- Validation input-contract exclusions: 4,209; unmatched cells: 1,433.
- DAVIS sequences without reliable timing remain excluded. One previously
  quarantined training header remains excluded by the spatial snapshot policy.

One additional supplied fake label is quarantined: `Figure Status Update - OpenAI
Speech-to-Speech Reasoning.mp4`. This is a physical robotics demonstration,
not a verified Sora-generated scene. AI control or speech synthesis does not make
camera imagery generated. The file and original frozen labels are untouched; it
is excluded only from this new motion experiment and listed in
`label_quarantine.json`. Other named Sora samples are retained. The reference is
the [official Figure upload](https://www.youtube.com/watch?v=Sq1QZB5baNw).

The usable train pool includes VideoUFO, KoNViD, Pexels, DeepAction families,
Kling, OpenAI Sora, GenVidBench pika, and small SVD/pika subsets. Not every stored
source participates: many sources are too low-FPS for this contract or have no
comparable other-class FPS/spatial cell. Nothing is deleted. See the complete
reasons and IDs in `motion_data_audit.json`. This deliberately trades some source
coverage for a more controlled experiment; it does not certify those excluded
generators will be detected.

## Learning and controls

The labels remain supervised real=0/fake=1. BCE with slight label smoothing updates
the backbone and classifier. The adapted input stem uses the classifier learning
rate (1e-4), while deeper blocks use 1e-5. A video-level label does not annotate where movement
is wrong. Providing velocity fields is an input constraint, not an object-physics
ground truth. Flow can still encode silhouettes, editing, blur, estimator errors,
and dataset-specific motion statistics. Natural-looking generated movement may
be indistinguishable by this branch alone; static videos contain little evidence.

Every completed epoch logs clean and flow-stress validation, precision/recall/FPR,
source metrics, confidence errors, group bootstrap intervals, coverage, gradients,
weight changes, throughput and VRAM. Stress means vector-field blur/quantization,
NOT video recompression. Early stopping uses the existing source-balanced average
of clean/stress scores, patience=5, maximum=12 epochs. It does not use test scores.

Three extra diagnostic controls are evaluated on the same validation IDs:

- `zero`: all motion channels zero; no RGB information can survive this input.
- `mean_field`: average velocity field repeated in time; removes evolution of
  the field but keeps its spatial structure and average motion.
- `shuffle`: permutes the estimated fields, preserving each field's values.

Their accuracies, changed decisions and score changes are logged and displayed in
the dashboard. Controls are NOT training labels or the early-stopping criterion.
An accuracy drop supports sensitivity to the removed signal, not proof of physical
understanding. If mean/shuffled fields behave like clean, temporal ordering may
not contribute even though the input contains motion.

The source-covered validation has a native-FPS confound: the old fixed metadata
rule "fake if FPS < 16" scores 66%, without seeing any pixels. This is NOT model
accuracy and the classifier does not receive FPS. Each motion report logs this
baseline and class-balanced quality within shared coarse FPS strata. Source
coverage and exact FPS/spatial balance cannot simultaneously be achieved with
an 800-video validation sample from the current pool. Strong results therefore
need per-source/FPS checks and an independent real-world test, not just aggregate
accuracy. The previous strictly matched 800-video validation had a 50% FPS-rule
baseline but almost all fake examples came from Pika; it is not used as the sole
selection benchmark.

The dashboard shows the reference RGB clip next to a color rendering of the
actual two-channel input. Hue represents direction and brightness speed; those
colors are not extra classifier channels. Grad-CAM uses layer2 for the shorter
motion input and is coarse/interpolated, not an exact trace of reasoning. Temporal
occlusion and fixed examples complement the global controls.

## Artifacts

All production artifacts are under `cache\temporal_v2\runs\motion_flow_v1`:

- `training.log`: detailed console/file log.
- `config.json`, `splits.json`, `motion_data_audit.json`, `data_audit.json`: protocol.
- `checkpoint.pt`: durable optimizer/scaler/RNG/sampler/clip-offset state.
- `best.pt`, `best.json`, `last.pt`: model weights and selection criteria.
- `epoch_NNN_plan.json`: exact selected video IDs and source counts.
- `epoch_NNN.json`, `history.json`: metrics and motion controls.
- `epoch_NNN_clean.csv`, `epoch_NNN_stress.csv`: per-video scores.
- `coverage.json`: unique optimized video IDs, not merely decoded files.
- `performance.jsonl`, `explanations_NNN.json`, `previews`: performance and images.
- `final_report.json`: once-only test/OOD evaluation of best weights, including
  explicit abstentions and all-requested accuracy bounds.

Checkpoints have a distinct `temporal-motion-v1` schema. At most 20 batches after
the last durable checkpoint replay following an interruption. There is no claim
of exact resume from an unsaved GPU instruction.

## Verification

Unit tests cover velocity units, translation removal, vector-aware augmentation,
absence of RGB reads in the dataset, the two-channel architecture, FPS matching,
low-FPS abstentions, protected RGB runs and exact checkpoint replay with BatchNorm.
`tests/temporal_flow_gpu_smoke.py` runs a separate eight-video epoch with real
files, validates finite parameters and updated weights/statistics, renders flow
and attribution, and checks original RGB model hashes before/after. It is a
technical smoke test, not an accuracy benchmark.

## References

- [OpenCV optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html):
  established dense motion estimator and its assumptions/limitations.
- [L3DE](https://arxiv.org/html/2406.19568v2): separate appearance, flow and depth
  cues for generated-video evaluation. This implementation uses a cheaper
  Farneback flow estimator, not their full RAFT/depth/contrastive pipeline.

After several full epochs, evaluate previously unused real camera videos and
modern generated videos. Decide whether this branch improves the spatial model
on the SAME IDs and whether its errors differ. Do not report ensemble improvements
without a held-out comparison or tune a fusion threshold on the final test set.
