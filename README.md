# osm_loc_bki

Global LiDAR localization against OpenStreetMap (no prior map, GPS or initial pose), and an OSM-BKI
semantic map built online at the poses it finds. ROS-free: C++ core, Python for maps and the live view.

## Setup

Run everything from the repository root.

```bash
conda env create -f environment.yaml      # env "osmlocbki"; -n NAME for another name
conda activate osmlocbki                  # (already have an env? activate it and run: pip install -e .)
export DATASET_PATH=/path/to/disk         # the disk holding datasets/mcd/...
```

Re-run `pip install -e .` after changing C++ code or moving the repository.

Expected data (MCD KTH shown; other sites in `config/sites.yaml`):

```
$DATASET_PATH/datasets/mcd/kth/
  hhs_calib.yaml                                        # lidar -> body extrinsic
  kth_day_09/
    pose_inW.csv                                        # poses
    lidar_bin/data/<scan>.bin                           # float32 x, y, z, intensity
    inferred_labels/<model>/predicted_labels/<scan>.bin # network labels (localization)
    inferred_labels/<model>/multiclass_alpha/<scan>.bin # network class distributions (mapping)
    gt_labels_terrain/<scan>.bin                        # ground truth, only for --eval
```

Ground-truth-label sites (`kth_gt`, `tuhh`, `ntu_gt`) read `<seq>/gt_labels/` and take no `--model`.

Maps for KTH, TUHH, NTU and CU are included. For a new site, add it to `config/sites.yaml`, then:

```bash
python -m osmlocbki.osm fetch   --site SITE   # data/maps/<map>.osm
python -m osmlocbki.osm convert --site SITE   # data/maps/<map>_{buildings,semantic}.geojson
```

The first run of a map or sequence builds and caches its database and descriptors in `data/`
(minutes for a new map); delete `data/db_<map>_*.npz` after changing a map.

## Run

```bash
python demos/run.py --site kth --seq kth_day_09 --model lidarrv_mcd                    # live view
python demos/run.py --site kth --seq kth_day_09 --model lidarrv_kitti360 --strength 2  # cross-domain model
python demos/run.py --site kth --seq kth_day_09 --model lidarrv_mcd \
    --no-view --eval gt_labels_terrain --save kth.png                                 # headless, scored
python demos/run.py --site tuhh --seq tuhh_day_03 --odometry drifting --scale-bias 0.02 --yaw-rw 0.5
```

| option | |
|---|---|
| `--strength S` | OSM strength; 0 = no OSM (default 0.1, in `config/mapping.yaml`; ~2 for cross-domain models) |
| `--pose lag\|filtered\|gt` | pose scans are mapped at: settled 60 m later (default), immediate, or ground truth |
| `--odometry near_exact\|drifting` | pose-graph odometry preset; `--scale-bias`, `--yaw-rw` simulate drift |
| `--max-kf N`, `--map-spacing M` | stop after N keyframes; metres between mapped scans (default 5) |
| `--eval DIR`, `--label-set SET` | score the map against `<seq>/DIR` labels (label set, default `mcd`) |
| `--no-view`, `--save PNG`, `--draw-every N` | headless; save the final view; keyframes between redraws |
| `--building-ids ...` | override the building label ids (normally from the label set) |

The view shows OSM, ground truth, the estimate and the last mapped scan (left), and the semantic map
from above with OSM building outlines (right). Close it to stop. From Python:

```python
import osmlocbki
p = osmlocbki.Pipeline("kth", "kth_day_09", model="lidarrv_mcd", strength=0.1)
while p.step():
    pass
p.finish()
p.trajectory("lag"), p.voxels()
```

## Configuration

- `config/sites.yaml`: datasets, their georeference and label set.
- `config/labels.yaml`: the map's classes, and each label set's mapping onto them (`label_sets:`),
  its building ids for localization (`localization_buildings:`) and which models use it (`models:`).
  A new model only needs a `models:` entry.
- `config/mapping.yaml`: mapping and OSM settings, including `strength` and the OSM confusion matrix
  (`osm_confusion/mcd_hand.yaml` by default).

## Tests

```bash
cmake -S . -B build -G Ninja && cmake --build build
./build/osmlocbki_tests [NAME]      # TUHH end-to-end tests run when DATASET_PATH holds MCD TUHH
python tests/test_python.py         # short KTH run when DATASET_PATH holds MCD KTH
```

They check localization against the original Python/GTSAM outputs, the mapper voxel-for-voxel against
the original OSM-BKI code (`tests/reference/make_reference.sh` regenerates that reference), and
TUHH day_03 end to end.

## How it works

Every 1 m, the scan's building points form a range descriptor matched against descriptors ray-cast
from OSM buildings; a rigid tracker, history check and sliding-window pose graph turn these matches
plus odometry into poses (`rigid`, `graph`, `filtered`, and `lag`, settled 60 m later). Every 5 m,
once the localizer is locked (100 m travelled and `rigid` agrees with `graph`), the scan is placed at
its pose and inserted into a Bayesian kernel inference voxel map: each point deposits the network's
class distribution, boosted toward the classes nearby OSM features support, gated by Patchwork++
ground segmentation.

Unlike the originals: OSM lines and points are buffered by their width (no libosmium); voxel
downsampling uses exact keys (OSM-BKI's hash merges mirror-image voxels); ROS, the OSM count prior,
free-space samples, the airborne-LiDAR surface and the offline smoother are dropped.

Limits: odometry, roll/pitch and height come from the dataset's ground truth (optionally with
simulated drift); the first ~100 m are not mapped.

## Credits

Descriptor and matching port the MATLAB of Y. Cho et al., "OpenStreetMap-Based LiDAR Global
Localization in Urban Environment Without a Prior LiDAR Map", RA-L 2022 (CC BY-NC-SA 4.0; derived
code carries the same terms). BKI mapping after L. Gan et al., RA-L 2020. Patchwork++
(`third_party/patchworkpp`, BSD-2), Urban Robotics Lab, KAIST. Map data © OpenStreetMap contributors
(ODbL).
