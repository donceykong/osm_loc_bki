# Airborne survey (DSM / DEM) in the voxel map

Can an airborne survey fill the map where the vehicle's LiDAR never looks (roofs, canopy, gaps), and
can its voxels be labelled? `project_dsm.py` runs the pipeline, voxelizes the survey within 50 m of
the mapped poses at the map's resolution, and compares ways to give those voxels a ground verdict
and a class (see its docstring).

```bash
# the survey model: data/surface/<map>_epsg<zone>.npz (dsm, dem, airborne + roof-edge wall points),
# built by osm_localization from LAS/LAZ tiles
PYTHONPATH=python python experiments/dsm/project_dsm.py --site kth --seq kth_day_09 --model lidarrv_mcd \
    --surface data/surface/mcd_kth_epsg32634.npz         # --reuse: skip the run, use the last one
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

## Next

- Insert survey voxels into the map as low-weight evidence (OSM + DEM ground deposit, so LiDAR
  overrides them), instead of a separate layer.
- Drop the vehicle class from OSM support at non-ground survey voxels over parking (canopy).
- CU main campus has a survey too (`data/surface/cu_main_epsg32613.npz`) but no mapped label set yet.
