"""The live 3D view of a run (pyvista): the OSM map draped over the ground, ground truth, the
estimate, the robot and its front-end hits, and the semantic map as it grows: its voxels (one point
each, class colours) and, with the nvblox backend, its TSDF mesh coloured by class. The camera
follows the newest settled (lag) pose, where the map is being built, and keeps whatever angle and
zoom you give it (drag, scroll).

Keys: v toggles the voxels, m the mesh, s the last mapped scan, g the ground (DEM <-> DSM).

The ground is the site's airborne survey (data/surface/<map>_epsg<zone>.npz) where there is one,
lifted onto the map's heights: its terrain (DEM), or its surface (DSM: roofs, canopy) with the OSM
draped over it; else a plane under the first mapped scan. Everything is drawn in the
map frame (UTM - map origin).
"""

import glob
import os

import numpy as np

from . import Site, data_dir

# osmloc's view colours, RGB 0..255: dark backdrop, OSM in greys and muted tints
BG = (28, 28, 28)
OSM_FILL = {"buildings": (70, 70, 70), "roads": (52, 52, 52), "sidewalks": (66, 56, 74), "cycleways": (52, 62, 72),
            "parking": (58, 58, 70), "grasslands": (44, 62, 44), "trees": (40, 58, 40), "forest": (36, 54, 36),
            "fences": (82, 64, 52)}
BUILDING_EDGE = (115, 115, 115)
GT, EST, HIT, ROBOT, TEXT = (90, 200, 90), (255, 150, 60), (255, 215, 0), (255, 255, 255), (235, 235, 235)
TILE = 50.0  # m: voxels are drawn in tiles, refreshed near each mapped scan


def rgb1(c):
    return tuple(v / 255 for v in c)


class Ground:
    """Terrain height in the map frame: the survey's DEM where the site has one, else a plane. surface()
    gives the DSM's heights instead (the DEM's where the DSM has none)."""

    def __init__(self, pipe):
        self.p, self.dem, self.dsm = pipe, None, None
        site = Site(pipe.site)
        hits = sorted(glob.glob(os.path.join(data_dir(), "surface", f"{site.map}_epsg*.npz")))
        if hits:
            with np.load(hits[0]) as z:
                self.dem, self.dsm, self.res = z["dem"], z["dsm"], float(z["res"])
                self.lo = z["origin"] - pipe.map_origin  # the DEM's corner in the map frame
        self.z0 = None  # plane height / datum shift, from the first mapped scan
        self.fallback = 0.0  # off the survey

    def ready(self):
        return self.z0 is not None

    def settle(self, pts):
        """Fix the plane (or the survey's datum) from a mapped scan's points (map frame)."""
        low = np.percentile(pts[:, 2], 5)
        if self.dem is None:
            self.z0 = float(low)
        else:
            off = self.p.survey_offset
            if not np.isfinite(off):  # no survey in the loop: its lowest points against the DEM
                d = pts[:, 2] - self._dem(pts[:, :2])
                off = float(np.nanpercentile(d, 5)) if np.isfinite(d).any() else 0.0
            self.z0 = off
            z = self._dem(pts[:, :2])
            self.fallback = float(np.nanmedian(z) + off) if np.isfinite(z).any() else float(low)

    def _dem(self, xy, grid=None):
        grid = self.dem if grid is None else grid
        ij = np.floor((xy - self.lo) / self.res).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < self.dem.shape[0]) & (ij[:, 1] < self.dem.shape[1])
        out = np.full(len(xy), np.nan, np.float32)
        out[ok] = grid[ij[ok, 0], ij[ok, 1]]
        return out

    def surface(self, xy):
        z = self(xy)
        if self.dsm is not None:
            s = self._dem(np.atleast_2d(xy), self.dsm) + self.z0
            z = np.where(np.isfinite(s), np.maximum(s, z), z)
        return z

    def __call__(self, xy):
        xy = np.atleast_2d(xy)
        if self.dem is None:
            return np.full(len(xy), self.z0)
        z = self._dem(xy) + self.z0
        z[~np.isfinite(z)] = self.fallback
        return z


def osm_texture(osm, origin, lo, hi, res):
    """The OSM map rasterized over [lo, hi] (map frame) at res m/px -> RGB image, row 0 north."""
    from PIL import Image, ImageDraw
    size = np.ceil((hi - lo) / res).astype(int)
    img = Image.new("RGB", tuple(int(s) for s in size), BG)
    draw = ImageDraw.Draw(img)
    px = lambda utm: [tuple(q) for q in ((np.asarray(utm) - origin) - [lo[0], hi[1]]) / [res, -res]]
    for cls, kind, xy in osm["features"]:
        if cls not in OSM_FILL or len(xy) < 2:
            continue
        if kind == "area" and len(xy) >= 3:
            draw.polygon(px(xy), fill=OSM_FILL[cls])
        else:
            draw.line(px(xy), fill=OSM_FILL[cls], width=max(1, int(round((3.0 if cls == "roads" else 1.5) / res))))
    for b in osm["buildings"]:
        draw.polygon(px(b), fill=OSM_FILL["buildings"], outline=BUILDING_EDGE)
    for h in osm["holes"]:
        draw.polygon(px(h), fill=BG, outline=BUILDING_EDGE)
    return np.asarray(img)


class LiveView3D:
    """Same use as view.LiveView: update() after steps, update(final=True) at the end, save(png)."""

    def __init__(self, pipe, output="lag", show=True, voxels=None, mesh=None, ground="dem", mesh_every=4,
                 point_size=3.0, cam_dist=160.0, width=1600, height=1000):
        import pyvista as pv
        self.pv, self.p, self.output = pv, pipe, output
        self.has_mesh = pipe.mapper == "nvblox"
        self.show_mesh = self.has_mesh if mesh is None else (mesh and self.has_mesh)
        self.show_voxels = (not self.show_mesh) if voxels is None else voxels
        self.show_scan, self.mesh_every, self.point_size = True, mesh_every, point_size
        tax = pipe.taxonomy()
        self.names, self.colors = tax["names"], np.asarray(tax["colors"], np.uint8)
        self.origin = np.asarray(pipe.map_origin)
        self.ground = Ground(pipe)
        self.dsm = ground == "dsm" and self.ground.dsm is not None
        if ground == "dsm" and self.ground.dsm is None:
            print(f"no survey DSM for {pipe.site}: the ground is a plane")
        self.texture = None
        self.n_mapped, self.n_meshed, self.tiles, self.closed = 0, 0, set(), False
        gt = np.asarray(pipe.gt) - self.origin
        self.lo, self.hi = gt.min(0) - 150, gt.max(0) + 150

        self.pl = pl = pv.Plotter(window_size=(width, height), off_screen=not show)
        pl.set_background(rgb1(BG))
        pl.add_legend([[self.names[c], self.colors[c] / 255] for c in range(1, len(self.names))]
                      + [["ground truth", rgb1(GT)], [f"estimate ({output})", rgb1(EST)], ["front-end hits", rgb1(HIT)]],
                      bcolor=rgb1(BG), border=False, face="rectangle", size=(0.12, 0.3), loc="upper right")
        pl.add_text(f"{pipe.site} / {pipe.seq}: OSM localization + OSM-BKI ({pipe.mapper} map)", position="upper_left",
                    font_size=11, color=rgb1(TEXT))
        self.robot = pl.add_mesh(pv.Sphere(radius=1.2), color=rgb1(ROBOT))
        self.gt_dot = pl.add_mesh(pv.Sphere(radius=0.7), color=rgb1(GT))
        p0 = np.r_[gt[0], 0.0]
        pl.camera_position = [tuple(p0 + np.array([-0.45, -0.55, 0.7]) * cam_dist), tuple(p0), (0, 0, 1)]
        for key, attr in (("v", "show_voxels"), ("m", "show_mesh"), ("s", "show_scan"), ("g", "dsm")):
            pl.add_key_event(key, lambda a=attr: self._toggle(a))
        if show:
            pl.show(interactive_update=True, auto_close=False)
        self.cam_set = False

    # ---- pieces ------------------------------------------------------------------------------
    def _toggle(self, attr):
        setattr(self, attr, not getattr(self, attr))
        if attr == "show_mesh" and not self.has_mesh:
            self.show_mesh = False
        if attr == "dsm":
            self.dsm = self.dsm and self.ground.dsm is not None
            if self.ground.ready():
                self._draw_ground()
        elif attr == "show_voxels":
            if self.show_voxels:
                self._voxels(None)
            else:
                for t in self.tiles:
                    self.pl.remove_actor(f"tile{t}")
                self.tiles.clear()
        elif attr == "show_mesh":
            if self.show_mesh:
                self._mesh(False)
            else:
                self.pl.remove_actor("mesh")
        elif attr == "show_scan" and not self.show_scan:
            self.pl.remove_actor("scan")
        self.pl.render()

    def _draw_ground(self):
        pv, g = self.pv, self.ground
        step = 1.0 if self.dsm else 2.0  # the DSM's buildings want the finer grid
        gx, gy = np.arange(self.lo[0], self.hi[0] + step, step), np.arange(self.lo[1], self.hi[1] + step, step)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        xy = np.c_[X.ravel(), Y.ravel()]
        Z = (g.surface(xy) if self.dsm else g(xy)).reshape(X.shape) - 0.3  # under the mapped ground
        surf = pv.StructuredGrid(X, Y, Z)
        surf.active_texture_coordinates = (surf.points[:, :2] - self.lo) / (self.hi - self.lo)
        if self.texture is None:
            self.texture = pv.Texture(osm_texture(self.p.osm(), self.origin, self.lo, self.hi, 0.5))
        # the DSM is lit, so its walls and roofs read as shapes; the flat DEM is drawn as is
        self.pl.add_mesh(surf, texture=self.texture, lighting=self.dsm, ambient=0.55, diffuse=0.6, name="ground",
                         reset_camera=False)

    def _path(self, xy, dz, color, name):
        ok = np.isfinite(xy[:, 0])
        if ok.sum() < 2:
            return
        xy = xy[ok]
        self.pl.add_mesh(self.pv.lines_from_points(np.c_[xy, self.ground(xy) + dz]), color=rgb1(color), line_width=3,
                         name=name, reset_camera=False)

    def _voxels(self, around):
        """(Re)draw the voxel tiles within reach of `around` (map frame xy), or all of them."""
        if around is None:
            vox = self.p.voxels()
        else:
            r = 70.0 + TILE
            vox = self.p.voxels(box=(around[0] - r, around[1] - r, around[0] + r, around[1] + r))
        xyz, cls = vox["xyz"], vox["cls"]
        if not len(cls):
            return
        tile = np.floor(xyz[:, :2] / TILE).astype(int)
        keys = tile[:, 0] * 100003 + tile[:, 1]
        order = np.argsort(keys, kind="stable")
        keys, xyz, cls, tile = keys[order], xyz[order], cls[order], tile[order]
        cut = np.flatnonzero(np.diff(keys)) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(keys)]):
            t = (int(tile[a, 0]), int(tile[a, 1]))
            if around is not None:  # only the tiles wholly inside the queried box are complete
                c = (np.array(t) + 0.5) * TILE
                if np.abs(c - around).max() > 70.0:
                    continue
            cloud = self.pv.PolyData(xyz[a:b].astype(np.float32))
            cloud["rgb"] = self.colors[np.clip(cls[a:b], 0, len(self.colors) - 1)]
            self.pl.add_mesh(cloud, scalars="rgb", rgb=True, point_size=self.point_size, style="points", lighting=False,
                             show_scalar_bar=False, reset_camera=False, name=f"tile{t}")
            self.tiles.add(t)

    def _mesh(self, full):
        m = self.p.mesh(full=full)
        v, f = m["vertices"], m["triangles"]
        if not len(f):
            return
        faces = np.c_[np.full(len(f), 3, np.int64), f].ravel()
        surf = self.pv.PolyData(v, faces)
        surf["rgb"] = m["colors"]
        surf.compute_normals(inplace=True, split_vertices=False)
        self.pl.add_mesh(surf, scalars="rgb", rgb=True, smooth_shading=True, show_scalar_bar=False, reset_camera=False,
                         name="mesh", ambient=0.35, diffuse=0.75)

    # ---- the loop ----------------------------------------------------------------------------
    def update(self, final=False):
        p, t = self.p, self.p.t
        if t < 0 or self.closed:
            return
        mapped = len(p.mapped)
        if mapped and not self.ground.ready():
            self.ground.settle(p.last_mapped()["points"])
            self._draw_ground()
        if self.ground.ready():
            est = p.trajectory(self.output) - self.origin
            self._path(np.asarray(p.gt)[: t + 1] - self.origin, 0.4, GT, "gt_path")
            self._path(est, 0.6, EST, "est_path")
            hits = np.asarray(p.hits(t)) - self.origin
            if len(hits):
                self.pl.add_mesh(self.pv.PolyData(np.c_[hits, self.ground(hits) + 0.8]), color=rgb1(HIT), point_size=9,
                                 render_points_as_spheres=True, name="hits", reset_camera=False)
            now = p.trajectory("filtered")[t] - self.origin
            here = np.r_[now, self.ground(now[None])[0] + 1.5]
            M = np.eye(4)
            M[:3, 3] = here
            self.robot.user_matrix = M
            g = np.asarray(p.gt)[t] - self.origin
            M = np.eye(4)
            M[:3, 3] = np.r_[g, self.ground(g[None])[0] + 1.0]
            self.gt_dot.user_matrix = M
            # follow the newest lag pose (the robot's until the first one settles), keeping the user's
            # angle and zoom
            lag = p.trajectory("lag")[: t + 1]
            ok = np.flatnonzero(np.isfinite(lag[:, 0]))
            if len(ok):
                at = lag[ok[-1]] - self.origin
                here = np.r_[at, self.ground(at[None])[0] + 1.5]
            cam = self.pl.camera
            off = np.array(cam.position) - np.array(cam.focal_point)
            cam.focal_point, cam.position = tuple(here), tuple(here + off)
            cam.clipping_range = (1, 8000)
        if mapped != self.n_mapped or final:
            s = p.last_mapped()
            if self.show_scan and mapped:
                keep = s["cls"] > 0
                cloud = self.pv.PolyData(s["points"][keep].astype(np.float32))
                cloud["rgb"] = self.colors[s["cls"][keep]]
                self.pl.add_mesh(cloud, scalars="rgb", rgb=True, point_size=1.5, style="points", lighting=False,
                                 show_scalar_bar=False, reset_camera=False, name="scan", opacity=0.6)
            if self.show_voxels and mapped:
                self._voxels(None if final else s["xy"] - self.origin)
            self.n_mapped = mapped
        if self.show_mesh and mapped and (final or mapped - self.n_meshed >= self.mesh_every or not self.n_meshed):
            self._mesh(final)
            self.n_meshed = mapped
        self.pl.add_text(p.status(), position="lower_left", font_size=9, color=rgb1(TEXT), name="hud")
        if self.pl.off_screen:
            self.pl.render()
        else:
            self.pl.update()
            if self.pl.render_window is None:
                self.closed = True

    def save(self, path):
        self.pl.screenshot(path)

    def hold(self):
        """Keep the window open (interactive) until it is closed."""
        if not self.pl.off_screen and not self.closed:
            self.pl.show(auto_close=False)
