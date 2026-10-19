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

**GPU mapping with nvblox (optional).** With a CUDA toolkit and the
[nvblox](https://github.com/nvidia-isaac/nvblox) submodule (`third_party/nvblox`, pinned at
`24eee49` and used unmodified), the build adds an nvblox backend (`--mapper nvblox`; the first build
takes a few minutes longer):

```bash
conda install -c conda-forge "cuda-version=13.0" cuda-nvcc cuda-cudart-dev cuda-cccl cuda-nvtx-dev libnpp-dev libcurand-dev
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init   # nvblox's LFS files (test data, images) are not needed
pip install -e .                          # python -c "import osmlocbki; print(osmlocbki.have_nvblox())"
```

Pick `cuda-version` for your driver (`nvidia-smi`) and host compiler (CUDA 13.0 takes the env's
gcc 15). `-Ccmake.define.NVBLOX_SOURCE_DIR=PATH` uses another checkout, `...OSMLOCBKI_NVBLOX=OFF`
skips it.

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
`cu_north_campus` (CU-Multi, sequence `robot1`) reads CENet's MCD-class labels from
`robot1/inferred_labels/cenet_mcd/` and also takes no `--model`; its georeference is a fit to the
robot's GPS (1.4 m median residual), so errors under ~1.5 m there are within the reference's noise.
`kitti360` takes `--seq 2013_05_28_drive_0000_sync` and a model (`lidarrv_kitti360`, `lidarrv_mcd`);
score it with `--eval gt_labels --label-set kitti360`. Its map is drive 0000's OSM extract (copy
`map_0000.osm` to `data/maps/kitti360_0000.osm`, then `convert --site kitti360`); its `lidar:` entry
sets the TSDF's range image for the HDL-64E.

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
| `--surface FILE\|auto` | project an airborne survey (DSM/DEM) into the map around each mapped pose (see below) |
| `--mapper cpu\|nvblox`, `--mesh PLY` | map backend (default `cpu`); nvblox: on the GPU with a TSDF, and its mesh coloured by class |
| `--view 2d\|3d`, `--show voxels\|mesh\|both` | matplotlib (default) or pyvista 3D view; what of the map it draws (default: the mesh with nvblox, else voxels) |
| `--building-ids ...` | override the building label ids (normally from the label set) |

The view shows OSM, ground truth, the estimate and the last mapped scan (left), and the semantic map
from above with OSM building outlines (right). Close it to stop. `--view 3d` (needs `pyvista`, in
`environment.yaml`) draws the OSM map draped over the ground (the site's survey terrain where there
is one, or with `--ground dsm` its surface: roofs and canopy), ground truth, the estimate, the robot
and its hits, and the map as it grows, the camera following the newest settled (lag) pose, where the
map is being built: drag and scroll to change its angle and zoom; `v` / `m` / `s` / `g` toggle the
voxels, the mesh, the last scan and the ground (DEM / DSM); the window stays open at the end.

```bash
python demos/run.py --site kth --seq kth_day_09 --model lidarrv_mcd --mapper nvblox --view 3d
```

From Python:

```python
import osmlocbki
p = osmlocbki.Pipeline("kth", "kth_day_09", model="lidarrv_mcd", strength=0.1)
while p.step():
    pass
p.finish()
p.trajectory("lag"), p.voxels()
p.mesh()        # nvblox: vertices, triangles, colours (map frame)
```

## Configuration

- `config/sites.yaml`: datasets, their georeference and label set.
- `config/labels.yaml`: the map's classes, and each label set's mapping onto them (`label_sets:`),
  its building ids for localization (`localization_buildings:`) and which models use it (`models:`).
  A new model only needs a `models:` entry.
- `config/mapping.yaml`: mapping and OSM settings, including `strength` and the OSM confusion matrix
  (`osm_confusion/mcd_hand.yaml` by default), the backend (`map: backend`), nvblox's LiDAR model and
  TSDF (`nvblox:`), and the airborne survey's (`survey:`).

**Airborne survey (optional).** With `--surface`, each mapped scan also projects the survey within
50 m into the map: every survey voxel once, labelled by OSM gated by its height over the survey's
terrain (ground or not), at a tenth of a LiDAR point's weight so LiDAR wins where it sees the same
place. This fills roofs, canopy and gaps the vehicle never sees. `auto` reads
`data/surface/<map>_epsg<zone>.npz` (KTH and CU main campus exist; they are built from LAS/LAZ tiles
or DSM/DEM rasters by osm_localization: `python -m osmloc.dsm --site S`, then
`osmloc build-surface --site S`). The height datum offset is estimated from each scan's ground points.

```bash
python demos/run.py --site kth --seq kth_day_09 --model lidarrv_mcd --surface auto
```

## Tests

```bash
cmake -S . -B build -G Ninja && cmake --build build
./build/osmlocbki_tests [NAME]      # TUHH end-to-end tests run when DATASET_PATH holds MCD TUHH
python tests/test_python.py         # short KTH run when DATASET_PATH holds MCD KTH
```

They check localization against the original Python/GTSAM outputs, the mapper voxel-for-voxel against
the original OSM-BKI code (`tests/reference/make_reference.sh` regenerates that reference), the nvblox
backend against the CPU one (and its free-space mask), and TUHH day_03 end to end.
`GLOG_minloglevel=1` quiets nvblox's logging.

## How it works

Every 1 m, the scan's building points form a range descriptor matched against descriptors ray-cast
from OSM buildings; a rigid tracker, history check and sliding-window pose graph turn these matches
plus odometry into poses (`rigid`, `graph`, `filtered`, and `lag`, settled 60 m later). Every 5 m,
once the localizer is locked (100 m travelled and `rigid` agrees with `graph`), the scan is placed at
its pose and inserted into a Bayesian kernel inference voxel map: each point deposits the network's
class distribution, boosted toward the classes nearby OSM features support, gated by Patchwork++
ground segmentation.

The nvblox backend keeps the same voxels in nvblox's GPU voxel blocks (the same evidence as the CPU
map, up to float rounding) and integrates every mapped scan into an nvblox TSDF on the same grid. A
voxel the TSDF sees as free space (`free_distance` in front of a surface) is left out of the map's
voxels, which drops the kernel's smear into the air and the trails of objects that moved; `--mesh`
writes the TSDF's mesh coloured by each voxel's class. Scoring (`--eval`) reads the evidence unmasked.
The TSDF takes every localization keyframe's scan (1 m apart, at the same settled poses), not only the
mapped ones, so the mesh is closed on the ground (`every_keyframe`; ~3 ms per keyframe). The blocks
live in CUDA unified memory (`unified_memory`), so a map larger than the GPU pages out to host RAM
instead of failing: CU north campus (3.3 km) holds 2.8 GB of blocks and fills the 8 GB card.

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
