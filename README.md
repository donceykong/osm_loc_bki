# OSM-LOC-BKI

Online, end to end: **global LiDAR localization against OpenStreetMap** (no prior LiDAR map, no
GPS, no initial pose) and an **OSM-BKI semantic map** built from the scans at the poses it finds.
ROS-free; C++ core, Python for OSM conversion and the live view.

It joins two codebases, keeping only what the method uses:

- **osm_localization** (osmloc): each scan's building points give a range descriptor (Cho et al.
  2022), matched against descriptors ray-cast from OSM building outlines; a back-end (rigid tracker,
  history check, sliding-window pose graph) turns those weak matches plus odometry into a trajectory.
- **OSM-BKI**: Bayesian kernel inference over voxels with Dirichlet class counts; each point deposits
  the segmentation network's class distribution, boosted toward the classes OSM supports there
  (distance falloff, confusion matrix, Patchwork++ ground gate).

## Quick start

All commands run from the repository root (`OSM-LOC-BKI/`).

### 1. Create the environment and install

```bash
conda env create -f environment.yaml           # env "osmlocbki": compilers, libraries, then pip install -e .
conda activate osmlocbki
```

The env name comes from the first line of `environment.yaml`; to use another, pass `-n`:
`conda env create -f environment.yaml -n my_env`. To install into an env you already have (it needs
the packages listed in `environment.yaml`): `conda activate my_env && pip install -e .`.

`pip install -e .` builds the C++ core with CMake and installs the `osmlocbki` package. After
changing C++ code, or after moving or copying the repository, run `pip install -e .` again (an
editable install points at the checkout it was installed from; `pip show -f osmlocbki` shows which).

### 2. Point at the datasets

```bash
export DATASET_PATH=/path/to/disk              # the disk holding datasets/mcd/...
```

Each site in `config/sites.yaml` names its folder under `$DATASET_PATH`. For MCD KTH:

```
$DATASET_PATH/datasets/mcd/kth/
  hhs_calib.yaml                                # lidar -> body extrinsic
  kth_day_09/
    pose_inW.csv                                # poses (ground truth: keyframes, odometry, scoring)
    lidar_bin/data/<scan>.bin                   # float32 x, y, z, intensity
    inferred_labels/<model>/predicted_labels/<scan>.bin   # int32, the network's labels
    inferred_labels/<model>/multiclass_alpha/<scan>.bin   # float32 Dirichlet alphas (the mapped scans)
    gt_labels_terrain/<scan>.bin                # int32 ground truth (only for --eval)
```

Sites with ground-truth labels (`kth_gt`, `tuhh`, `ntu_gt`) read `<seq>/gt_labels/` instead and need no
model.

### 3. Prepare the OSM map (once per map)

```bash
python -m osmlocbki.osm fetch   --site kth     # data/maps/<map>.osm from Overpass (skip if present)
python -m osmlocbki.osm convert --site kth     # -> data/maps/<map>_buildings.geojson, <map>_semantic.geojson
```

The KTH, TUHH, NTU and CU maps are already in `data/maps/`. The localization database
(`data/db_<map>_free2_n360_r50.npz`) and each sequence's scan descriptors (`data/queries_*.npz`) are
built on the first run and cached; a new map's database takes a few minutes. Delete a database after
changing its map.

### 4. Run

```bash
# live view: the network's building points localize, its class distributions map
python demos/run.py --site kth --seq kth_day_09 --model lidarrv_mcd

# headless: score the map against ground-truth labels and save the final view
python demos/run.py --site kth --seq kth_day_09 --model lidarrv_mcd --no-view --eval gt_labels_terrain --save kth.png

# a network from another domain (KITTI-360 labels), with a strong OSM prior
python demos/run.py --site kth --seq kth_day_09 --model lidarrv_kitti360 --strength 2

# ground-truth labels (an oracle input), drifting odometry
python demos/run.py --site tuhh --seq tuhh_day_03 --odometry drifting --scale-bias 0.02 --yaw-rw 0.5
```

Options (`python demos/run.py --help`):

| option | |
|---|---|
| `--pose lag\|filtered\|gt` | the pose scans are mapped at: settled 60 m later (default), at once, or ground truth (an oracle) |
| `--strength S` | OSM strength (default: `config/mapping.yaml`, 0.1) |
| `--odometry near_exact\|drifting` | the graph's odometry preset; `--scale-bias`, `--yaw-rw` simulate drift |
| `--max-kf N` | stop after N localization keyframes |
| `--map-spacing M` | metres between mapped keyframes (default 5) |
| `--building-ids ...` | override the localizer's building label ids (default: the label set's, `config/labels.yaml`) |
| `--eval DIR` / `--label-set SET` | score the map against `<seq>/DIR` labels in label set SET (default mcd) |
| `--no-view`, `--save PNG`, `--draw-every N` | headless, save the final view, keyframes between redraws |

It prints progress every 100 keyframes, then each output's share of keyframes within 5 m of ground
truth and, with `--eval`, the map's per-class IoU. The view: left, the OSM map, ground truth, the
estimate, the newest front-end hits and the last mapped scan at its pose; right, the semantic map from
above around that scan (the whole map at the end) with the OSM building outlines on top. Close the
window to stop early.

From Python:

```python
import osmlocbki
p = osmlocbki.Pipeline("kth", "kth_day_09", model="lidarrv_mcd", pose="lag")
while p.step():
    pass
p.finish()
p.trajectory("lag"), p.voxels(), p.classify_keyframe(p.mapped[0])
```

### 5. Test

```bash
# C++ tests (pip does not build them)
cmake -S . -B build -G Ninja && cmake --build build
./build/osmlocbki_tests                          # with DATASET_PATH holding MCD TUHH: also the TUHH runs

# Python: the OSM converter, and with DATASET_PATH holding MCD KTH a short run and the view
python tests/test_python.py
```

`./build/osmlocbki_tests NAME` runs the tests whose name contains NAME.

### Without installing

Build in place and import from the source tree (the module lands in `python/osmlocbki/`):

```bash
cmake -S . -B build -G Ninja && cmake --build build
export PYTHONPATH=$PWD/python
```

Use one or the other: with the package installed, unset `PYTHONPATH` (or delete
`python/osmlocbki/_osmlocbki*.so`) so Python loads the installed module.

## How it works

Per **localization keyframe** (every 1 m), using nothing after it:

| output | |
|---|---|
| `rigid` | the best rigid placement of the dead-reckoned trajectory, scored on all keyframes so far |
| `graph` | a 200 m sliding window of poses with odometry and robust hit priors, seeded by `rigid` |
| `filtered` | odometry carries the pose; each keyframe it moves 15% of the way to `graph` (gaps over 8 m at once) |
| `lag` | each pose from the window 60 m of travel later |

Per **mapping keyframe** (every 5 m, among the scans the network exported class distributions for),
once its pose is ready and the localizer is **locked** (100 m of path behind, `rigid` and `graph`
agree): the scan is levelled (roll and pitch from the dataset's pose, as an IMU would give), placed
at the pose (`--pose lag`, the default; `filtered` maps at once; `gt` is an oracle baseline),
ground-segmented with Patchwork++, and inserted:

- each point's OSM support per OSM class: 1 inside, falling linearly to 0 at `decay_m` past the edge
  (roads, sidewalks, fences, single trees are bands of their width), plus "none"
- `s = M w` with the confusion matrix `M = P(class | OSM class)`; classes whose ground role
  (`config/labels.yaml`) contradicts the point's Patchwork++ verdict are dropped
- the point deposits `y = p + strength * s * (1 - p)` through the sparse kernel, length
  `ell * a * exp(1 + b * vacuity)`, into the voxels of its block and the 6 neighbouring blocks

Settings: `config/mapping.yaml` (the paper's; `strength` 0.1 for an in-domain model, ~2 under
domain shift), `config/labels.yaml` (taxonomy, ground roles, label sets), `config/sites.yaml`.

**Label sets.** Every run's labels are in a label set (`config/labels.yaml`): a site's own for its
ground-truth labels (`label_set:` in `config/sites.yaml`), the model's (`models:`) for a network's.
The set maps raw label ids onto the common classes (the map's input) and names the localizer's
building points (`localization_buildings:`, the ids matching OSM building outlines; by default every
id mapped to `building`), so a model with another label set needs no other setting. A new model: add
it under `models:`; a new label set: add its mapping.

**OSM strength.** `strength` in `config/mapping.yaml` (default 0.1), `--strength S` on
`demos/run.py`, or `Pipeline(..., strength=S)`; 0 turns OSM off.

## What the tests check

- localization against the original Python implementation and GTSAM (`tests/data`, from osmloc)
- OSM support, the ground gate, the kernel and the OSM boost, analytically
- the semantic map against the **original OSM-BKI mapper** on a synthetic scan (every voxel's
  Dirichlet counts within 3e-6): `tests/reference/make_reference.sh` rebuilds the reference from an
  OSM-BKI checkout (needs PCL, not ROS)
- end to end on MCD TUHH day_03: osmloc's README row (rigid 93.9%, lag 99.1% within 5 m), and a map
  built at the lag poses reproducing the ground-truth-pose map

## Differences from the originals

- **OSM lines and points are bands** (distance to the centreline minus half the width) instead of
  OSM2World-style band polygons with junction cuts; tag-to-class rules follow OSM-BKI's, in
  `python/osmlocbki/osm.py` (no libosmium).
- **Voxel downsampling keys voxels exactly.** OSM-BKI's hash `(ix*73856093)^(iy*19349669)^(iz*83492791)`
  gives mirror-image voxels ((x, y, z) and (-x, -y, z)) the same key, averaging their points into a
  bogus training point between them (about 0.35% of voxels on the reference scan).
- Points within 0.5 m of the sensor (zero-filled non-returns) are dropped before mapping.
- Not carried over: ROS/RViz, the OSM Dirichlet count prior (an ablation, off by default), free-space
  beam samples (they deposit no evidence), the airborne-LiDAR surface, the offline smoother, the
  descriptor refiner and road-sampled candidates.

## Limits

- Odometry, roll/pitch and height come from the dataset's ground truth (with optional simulated drift).
- Keyframes before lock (~100 m) are not mapped; on a large map with weak evidence the localizer can
  lock onto a look-alike place, and the map follows it.

## Credits

The descriptor and two-stage matching port the MATLAB of Y. Cho, G. Kim, S. Lee, J.-H. Ryu,
"OpenStreetMap-Based LiDAR Global Localization in Urban Environment Without a Prior LiDAR Map",
IEEE RA-L 7(2), 2022, licensed CC BY-NC-SA 4.0; derived code carries the same terms. BKI mapping
after L. Gan et al., "Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping",
RA-L 2020. Patchwork++ (`third_party/patchworkpp`, BSD-2) by the Urban Robotics Lab, KAIST.
Map data © OpenStreetMap contributors (ODbL).
