// Python bindings: sites, and the end-to-end pipeline with everything the live view draws.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "osmloc/site.hpp"
#include "osmlocbki/pipeline.hpp"

namespace py = pybind11;
using namespace olb;

namespace {

MatX2 mat(const std::vector<Vec2>& r) {
  MatX2 m(r.size(), 2);
  for (size_t i = 0; i < r.size(); ++i) m.row(i) = r[i].transpose();
  return m;
}

std::unique_ptr<Pipeline> make_pipeline(const std::string& site, const std::string& seq, const std::string& model,
                                        const std::string& pose, bool map, double strength, const std::string& odometry,
                                        double scale_bias, double yaw_rw, unsigned seed, long max_kf, double spacing,
                                        double map_spacing, double lock_m, double lag_m, std::vector<int> building_ids,
                                        const std::string& mapping, const std::string& surface,
                                        const std::string& mapper, bool loc_surface, double lock_agree_m) {
  PipelineConfig c;
  osmloc::RunConfig& l = c.loc;
  l.site = site, l.seq = seq, l.model = model;
  if (odometry != "near_exact" && odometry != "drifting") throw std::invalid_argument("odometry: near_exact or drifting");
  l.graph = odometry == "drifting" ? osmloc::GraphParams::drifting() : osmloc::GraphParams();
  l.odometry.scale_bias = scale_bias, l.odometry.yaw_rw = yaw_rw, l.odometry.seed = seed;
  l.max_kf = max_kf, l.spacing = spacing, l.lag_m = lag_m, l.building_ids = std::move(building_ids);
  l.surface = loc_surface;
  c.pose = pose, c.map = map, c.strength = strength, c.map_spacing = map_spacing, c.lock_m = lock_m;
  c.lock_agree_m = lock_agree_m;
  c.mapping = mapping, c.surface = surface, c.mapper = mapper;
  return std::make_unique<Pipeline>(c);
}

py::dict osm_layers(const Pipeline& p) {
  const OsmMap& m = p.osm();
  py::list buildings, holes, features;
  for (const auto& b : m.buildings) {
    buildings.append(mat(b.outer));
    for (const auto& h : b.holes) holes.append(mat(h));
  }
  for (const auto& s : m.features)
    features.append(py::make_tuple(osm_class_name(s.cls), s.is_area() ? "area" : s.path.size() == 1 ? "point" : "line",
                                   s.is_area() ? mat(s.area.outer) : mat(s.path)));
  return py::dict(py::arg("epsg") = m.epsg, py::arg("buildings") = buildings, py::arg("holes") = holes,
                  py::arg("features") = features);
}

py::dict voxels(const Pipeline& p, std::optional<Box> box) {
  if (!p.semantic_map()) throw std::logic_error("no map: the pipeline runs with map=False");
  MatX3f xyz;
  std::vector<int> cls;
  std::vector<float> prob, mass;
  {
    py::gil_scoped_release free;
    p.semantic_map()->voxels(xyz, cls, prob, mass, box ? &*box : nullptr);
  }
  return py::dict(py::arg("xyz") = xyz, py::arg("cls") = py::array(py::cast(cls)), py::arg("prob") = py::array(py::cast(prob)),
                  py::arg("mass") = py::array(py::cast(mass)), py::arg("origin") = p.semantic_map()->utm_origin());
}

py::dict last_mapped(const Pipeline& p) {
  const MappedScan& s = p.last_mapped();
  return py::dict(py::arg("f") = s.f, py::arg("scan_id") = s.scan_id, py::arg("xy") = s.xy, py::arg("psi") = s.psi,
                  py::arg("points") = s.points, py::arg("cls") = py::array(py::cast(s.cls)));
}

py::dict taxonomy(const Pipeline& p) {
  const Taxonomy& t = p.taxonomy();
  py::array_t<uint8_t> colors({static_cast<long>(t.size()), 3L});
  for (int c = 0; c < t.size(); ++c)
    for (int k = 0; k < 3; ++k) colors.mutable_at(c, k) = t.colors[c][k];
  return py::dict(py::arg("names") = t.names, py::arg("colors") = colors);
}

// OSM's gated class support at UTM points (N, 2) with ground verdicts (N: 1, 0, -1) -> (N, C)
MatXf osm_support(const Pipeline& p, const MatX2& xy, const std::vector<int8_t>& ground) {
  if (!p.osm_prior()) throw std::logic_error("no OSM prior: the mapping config has no confusion matrix");
  if (static_cast<long>(ground.size()) != xy.rows()) throw std::invalid_argument("one ground verdict per point");
  MatXf s(xy.rows(), p.osm_prior()->num_class());
  py::gil_scoped_release free;
#pragma omp parallel for schedule(static)
  for (long i = 0; i < xy.rows(); ++i) p.osm_prior()->project(xy.row(i).transpose(), ground[i], s.row(i).data());
  return s;
}

MatX2 hits(const Pipeline& p, long t) {
  const auto& b = p.localizer().buf();
  MatX2 h(b.k, 2);
  for (int j = 0; j < b.k; ++j) h.row(j) = b.e_xy(static_cast<int>(t), j).transpose();
  return h;
}

}  // namespace

PYBIND11_MODULE(_osmlocbki, m) {
  m.doc() = "Global localization against OpenStreetMap and OSM-BKI semantic mapping (C++)";

  m.def("sites", &Site::names, "Site names (osm_localization's config/sites.yaml).");
  m.def("have_nvblox", &have_nvblox, "Whether this build has the nvblox (GPU) map backend.");
  m.def("data_dir", &data_dir, "Where generated files go (data/ or $OSMLOCBKI_DATA).");
  m.def("config_dir", &config_dir, "The repository's config/.");
  m.def(
      "segment_ground",
      [](const MatXf& pts, double sensor_height, double min_range, double max_range) {
        GroundParams g;
        g.sensor_height = sensor_height, g.min_range = min_range, g.max_range = max_range;
        py::gil_scoped_release free;
        return GroundSegmenter(g).segment(pts);
      },
      py::arg("points"), py::arg("sensor_height") = 1.73, py::arg("min_range") = 0.0, py::arg("max_range") = 200.0,
      "Patchwork++ on one levelled cloud (N, 3), sensor at the origin -> per point 1 ground, 0 non-ground, -1 unknown.");
  m.def("label_mapping", [](const std::string& set) { return Taxonomy::load().mapping(set); }, py::arg("label_set"),
        "Raw label id -> common class (index = raw id) for a label set of config/labels.yaml.");

  py::class_<Site>(m, "Site")
      .def(py::init(&Site::load), py::arg("name"))
      .def_readonly("name", &Site::name)
      .def_readonly("map", &Site::map)
      .def_property_readonly("map_bbox", [](const Site& s) { return s.loc().map_bbox; })
      .def("seq_dir", [](const Site& s, const std::string& seq) { return s.loc().seq_dir(seq); }, py::arg("seq"))
      .def("label_set", &Site::label_set, py::arg("model") = "");

  py::class_<Pipeline>(m, "Pipeline")
      .def(py::init(&make_pipeline), py::arg("site"), py::arg("seq"), py::arg("model") = "", py::arg("pose") = "lag",
           py::arg("map") = true, py::arg("strength") = -1.0, py::arg("odometry") = "near_exact",
           py::arg("scale_bias") = 0.0, py::arg("yaw_rw") = 0.0, py::arg("seed") = 1u, py::arg("max_kf") = 0,
           py::arg("spacing") = 1.0, py::arg("map_spacing") = 5.0, py::arg("lock_m") = 100.0, py::arg("lag_m") = 60.0,
           py::arg("building_ids") = std::vector<int>{}, py::arg("mapping") = "mapping.yaml", py::arg("surface") = "",
           py::arg("mapper") = "", py::arg("loc_surface") = false, py::arg("lock_agree_m") = 2.0,
           "Localize a sequence keyframe by keyframe (osm_localization) and map it with OSM-BKI (see pipeline.hpp).\n"
           "pose: lag | filtered | gt; strength: OSM strength (< 0: config/mapping.yaml's); odometry: near_exact or\n"
           "drifting graph preset, with simulated drift scale_bias (fraction) and yaw_rw (deg / sqrt(m)).\n"
           "surface: an airborne survey .npz projected into the map around each mapped pose ('auto':\n"
           "data/surface/<map>_epsg<zone>.npz). mapper: cpu | nvblox (GPU, with a TSDF; '': config/mapping.yaml's).\n"
           "loc_surface: osm_localization registers against the site's airborne survey (its sites.yaml surface:).")
      .def("step", &Pipeline::step, py::call_guard<py::gil_scoped_release>(),
           "Localize the next keyframe and map what is ready -> False when done.")
      .def("finish", &Pipeline::finish, py::call_guard<py::gil_scoped_release>(), "Settle the lag and map what it settles.")
      .def("status", &Pipeline::status)
      .def_property_readonly("site", [](const Pipeline& p) { return p.config().loc.site; })
      .def_property_readonly("seq", [](const Pipeline& p) { return p.config().loc.seq; })
      .def_property_readonly("map_origin", [](const Pipeline& p) { return p.semantic_map()->utm_origin(); },
                             "UTM of the map frame's origin.")
      .def_property_readonly("t", &Pipeline::t)
      .def_property_readonly("size", &Pipeline::size)
      .def_property_readonly("locked", &Pipeline::locked)
      .def_property_readonly("gt", &Pipeline::gt, "The keyframes' LiDAR ground-truth positions, UTM (T, 2).")
      .def_property_readonly("gt_psi", &Pipeline::gt_psi)
      .def_property_readonly("scan_ids", &Pipeline::keyframe_ids, "The keyframes' scan ids.")
      .def_property_readonly("frame_scan_ids", [](const Pipeline& p) { return p.frames().scan_id; },
                             "The scan id of each frame (the scans placed off keyframes; mapped indexes them).")
      .def_property_readonly("dist", [](const Pipeline& p) { return p.localizer().buf().dist; },
                             "Odometry path length per localized keyframe.")
      .def("trajectory", [](const Pipeline& p, const std::string& o) { return p.localizer().trajectory(o); },
           py::arg("output"), "(n, 2) UTM positions of rigid | graph | hybrid | lag (NaN until settled) | filtered.")
      .def("heading", [](const Pipeline& p, const std::string& o) { return p.localizer().heading(o); }, py::arg("output"))
      .def("hits", &hits, py::arg("t"), "Keyframe t's front-end hits (k, 2).")
      .def_property_readonly("map_frames", &Pipeline::map_frames, "The frames to map.")
      .def_property_readonly("mapped", &Pipeline::mapped, "The frames mapped so far.")
      .def_property_readonly("mapped_xy", [](const Pipeline& p) { return mat(p.mapped_xy()); },
                             "Where each mapped scan was mapped, UTM (n, 2).")
      .def("last_mapped", &last_mapped, "The last mapped scan: f (frame), scan_id, xy, psi, points (map frame), cls.")
      .def("voxels", &voxels, py::arg("box") = std::nullopt,
           "Voxels with evidence (within box = (xmin, ymin, xmax, ymax), map frame): xyz (map frame), cls, prob,\n"
           "mass, origin (UTM of the map frame).")
      .def("classify",
           [](const Pipeline& p, const MatX3f& pts) {
             if (!p.semantic_map()) throw std::logic_error("no map: the pipeline runs with map=False");
             return p.semantic_map()->classify(pts);
           },
           py::arg("points"), "The class of the voxel holding each point (N, 3), map frame (0: free or unobserved).")
      .def("osm_support", &osm_support, py::arg("xy"), py::arg("ground"),
           "OSM's ground-gated support per class at UTM points (N, 2), ground verdicts (N,) -> (N, C).")
      .def("classify_frame", &Pipeline::classify_frame, py::arg("f"),
           "Frame f's scan at its ground-truth pose, classified by the map (one class per raw point).")
      .def("osm", &osm_layers, "The OSM map in UTM: buildings, holes, features [(class, area|line|point, (N, 2))].")
      .def("taxonomy", &taxonomy, "The common classes' names and colours.")
      .def("save_mesh",
           [](const Pipeline& p, const std::string& path) {
             if (!p.semantic_map()) throw std::logic_error("no map: the pipeline runs with map=False");
             py::gil_scoped_release free;
             p.semantic_map()->save_mesh(path, p.taxonomy().colors);
           },
           py::arg("path"), "The nvblox backend's TSDF mesh, coloured by class, as PLY.")
      .def("mesh",
           [](const Pipeline& p, bool full) {
             if (!p.semantic_map()) throw std::logic_error("no map: the pipeline runs with map=False");
             MatX3f v;
             MatX3i f;
             MatX3u8 c;
             {
               py::gil_scoped_release free;
               p.semantic_map()->mesh(v, f, c, p.taxonomy().colors, full);
             }
             py::dict d;
             d["vertices"] = v, d["triangles"] = f, d["colors"] = c, d["origin"] = p.semantic_map()->utm_origin();
             return d;
           },
           py::arg("full") = false,
           "The nvblox backend's TSDF mesh coloured by class, map frame: vertices (N, 3), triangles (M, 3), colors\n"
           "(N, 3) uint8, origin (UTM of the map frame). full=False re-meshes only what changed since the last call.")
      .def("memory", [](const Pipeline& p) { return p.semantic_map() ? p.semantic_map()->memory() : std::string(); },
           "The nvblox backend's GPU memory per layer.")
      .def_property_readonly("mapper", [](const Pipeline& p) { return p.semantic_map() ? p.semantic_map()->params().backend : ""; })
      .def_property_readonly("map_seconds", &Pipeline::map_seconds)
      .def_property_readonly("survey_voxels", &Pipeline::survey_voxels, "Survey voxels projected into the map so far.")
      .def_property_readonly("survey_offset", &Pipeline::survey_offset, "Survey height -> map z (NaN until estimated).");
}
