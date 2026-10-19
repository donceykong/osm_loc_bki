# Airborne survey (DSM / DEM) in the voxel map

Can an airborne survey fill the map where the vehicle's LiDAR never looks (roofs, canopy, gaps), and
can its voxels be labelled? `project_dsm.py` runs the pipeline, voxelizes the survey within 50 m of
the mapped poses at the map's resolution, and compares ways to give those voxels a ground verdict
and a class (see its docstring).

```bash
# the survey model: $DATASET_PATH/<site data_dir>/surface/<map>_epsg<zone>.npz (dsm, dem, airborne + roof-edge wall points),
# built by osm_localization from LAS/LAZ tiles
PYTHONPATH=python python experiments/dsm/project_dsm.py --site kth --seq kth_day_09 --model lidarrv_mcd \
    --surface $DATASET_PATH/datasets/mcd/kth/surface/mcd_kth_epsg32634.npz         # --reuse: skip the run, use the last one
```

Writes `out/<site>_<seq>_survey.png` (six top-down panels) and `out/<site>_<seq>_survey_voxels.npz`.

## KTH day_09, lidarrv_mcd, lag poses (2026-10-13)

- **Datum.** The survey's heights and the map's agree to 6 cm (median over LiDAR ground-class voxels
  against the DEM; 10-90% spread ±0.9 m, mostly DEM smoothing), so no vertical correction is needed here.
- **Ground verdicts**, against the ground role of the overlapping LiDAR voxel's class (195k voxels):
  height above the DEM (< 0.4 m) agrees 96.2%, Patchwork++ on the survey (virtual sensor at each pose)
  95.3%. Both work; the DEM is simpler and slightly better.
- **Labels without LiDAR** (OSM support gated by the DEM verdict) agree with the LiDAR map on 88% of
  overlapping voxels: building 97%, vegetation 85%, terrain 78%, but road 51%, sidewalk 40%,
  parking 16% (OSM's bands and the road/sidewalk split are coarse). Canopy over parking lots comes out
  as vehicle (the ground gate drops parking there; the confusion matrix sends parking to vehicle).
- **Coverage** within 50 m of the mapped poses (1 m columns): LiDAR 4.38 ha, survey 5.08 ha, both
  5.29 ha (+21%); 0.62 ha of it is roof or canopy above 3 m the LiDAR map does not have.
- **LiDAR labels reaching the survey** (nearest LiDAR voxel within 2 voxels) cover 65% of survey
  voxels; the rest (mostly roofs and far canopy) need OSM + the ground verdict.

## In the loop

`demos/run.py --surface auto` (see the main README) projects the survey online with the DEM ground
verdict, the better of the two: KTH day_09 gets 414k survey voxels, the datum estimate settles at
-0.04 m (here -0.06 m offline), and the map's score on the LiDAR scans is unchanged (mIoU 0.381 with
and without; LiDAR outweighs the survey where both are), at ~55 ms more per mapped scan.

## From rasters (KITTI-360)

Sites whose survey comes as GeoTIFFs build the same model with `raster_survey.py` (needs rasterio,
pyproj, pyyaml; not in `environment.yaml`): the DSM's surface every 0.25 m plus wall points at the
edges of building-like cells, cropped to the site's `map_bbox` + 100 m.

```bash
G=$DATASET_PATH/datasets/kitti360/geospatial/kitti360_DGM025-2026
python experiments/dsm/raster_survey.py --site kitti360 \
    --dsm $G/kitti360_DOM1/merged_dsm.tif --dem $G/kitti360_DGM025/merged_dem.tif
# -> $DATASET_PATH/datasets/kitti360/surface/k360_0000_epsg32632.npz (1.2 x 1.3 km, 25.7M points, 366 MB; 5 s)
```

KITTI-360 drive 0000 (2026-10-18): the rasters (ETRS89 / UTM 32N, DHHN2016 heights) need no
correction. LiDAR points above the sensor at ground-truth poses fall on elevated DSM cells most at
zero shift (82.6%, falling off evenly within +-2 m), and the sensor sits a median 1.62 m over the
DEM (1.73 m nominal); in the loop the datum settles at -0.14..-0.22 m. Over the first 500 keyframes
(lidarrv_kitti360) the map scores mIoU 0.619 with and without the survey's ~800k voxels.

## Next

- Drop the vehicle class from OSM support at non-ground survey voxels over parking (canopy).
- CU main campus has a survey too (`datasets/CU_MULTI/main_campus/surface/cu_main_epsg32613.npz`) but no mapped label set yet.
