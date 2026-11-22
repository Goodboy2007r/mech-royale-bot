"""Static world model for mechfight/1: terrain, solids, vision/shot rays, nav grids, A*.

The map is the same every match (protocol.md section 11).  Construction is done once at `hello`;
after that every structure is read-only, so the bot can query it cheaply every tick.
"""
from __future__ import annotations

import json
import math
import os
import time
from array import array
from heapq import heappush, heappop
from typing import Dict, List, Optional, Sequence, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


def _bilinear(grid: array, nx: int, ny: int, x: float, y: float) -> float:
    if x < 0:
        x = 0.0
    elif x > nx - 1.000001:
        x = nx - 1.000001
    if y < 0:
        y = 0.0
    elif y > ny - 1.000001:
        y = ny - 1.000001
    i = int(x)
    j = int(y)
    fx = x - i
    fy = y - j
    k = j * nx + i
    a = grid[k]
    b = grid[k + 1]
    c = grid[k + nx]
    d = grid[k + nx + 1]
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy


def _point_in_ring(px: float, py: float, ring: Sequence[float]) -> bool:
    inside = False
    n = len(ring)
    jx = ring[n - 2]
    jy = ring[n - 1]
    for i in range(0, n, 2):
        ix = ring[i]
        iy = ring[i + 1]
        if (iy > py) != (jy > py):
            xcross = (jx - ix) * (py - iy) / (jy - iy) + ix
            if px < xcross:
                inside = not inside
        jx, jy = ix, iy
    return inside


def _ring_bbox(ring: Sequence[float]) -> Tuple[float, float, float, float]:
    xs = ring[0::2]
    ys = ring[1::2]
    return min(xs), min(ys), max(xs), max(ys)


class Prism:
    __slots__ = ("kind", "base", "top", "ring", "holes", "bbox", "hbboxes")

    def __init__(self, p: dict):
        self.kind = p.get("kind", "building")
        self.base = float(p.get("base", 0.0))
        self.top = float(p.get("top", 0.0))
        self.ring = p["ring"]
        self.holes = p.get("holes") or []
        self.bbox = _ring_bbox(self.ring)
        self.hbboxes = [_ring_bbox(h) for h in self.holes]

    def contains_xy(self, x: float, y: float) -> bool:
        b = self.bbox
        if x < b[0] or x > b[2] or y < b[1] or y > b[3]:
            return False
        if not _point_in_ring(x, y, self.ring):
            return False
        for h, hb in zip(self.holes, self.hbboxes):
            if hb[0] <= x <= hb[2] and hb[1] <= y <= hb[3] and _point_in_ring(x, y, h):
                return False
        return True


class NavGrid:
    """Coarse 2.5D walkability grid with A* and smoothing."""

    def __init__(self, world: "World", res: float):
        self.world = world
        self.res = float(res)
        b = world.bounds
        self.minx = float(b["minX"])
        self.miny = float(b["minY"])
        self.nx = int(math.ceil((b["maxX"] - b["minX"]) / self.res)) + 1
        self.ny = int(math.ceil((b["maxY"] - b["minY"]) / self.res)) + 1
        self.blocked = bytearray(self.nx * self.ny)
        self.solid = bytearray(self.nx * self.ny)
        self._build()

    def _center(self, ix: int, iy: int) -> Vec2:
        return self.minx + (ix + 0.5) * self.res, self.miny + (iy + 0.5) * self.res

    def cell_of(self, x: float, y: float) -> Tuple[int, int]:
        ix = int((x - self.minx) / self.res)
        iy = int((y - self.miny) / self.res)
        if ix < 0:
            ix = 0
        elif ix >= self.nx:
            ix = self.nx - 1
        if iy < 0:
            iy = 0
        elif iy >= self.ny:
            iy = self.ny - 1
        return ix, iy

    def _build(self) -> None:
        w = self.world
        nx, ny, res = self.nx, self.ny, self.res
        blocked = self.blocked
        solid = self.solid
        step_up = 1.2
        mech_h = 9.0
        # Rasterise each prism's bounding box only once.
        for p in w.prisms:
            b = p.bbox
            ix0 = max(0, int((b[0] - self.minx) / res) - 1)
            ix1 = min(nx - 1, int((b[2] - self.minx) / res) + 1)
            iy0 = max(0, int((b[1] - self.miny) / res) - 1)
            iy1 = min(ny - 1, int((b[3] - self.miny) / res) + 1)
            for iy in range(iy0, iy1 + 1):
                y = self.miny + (iy + 0.5) * res
                if y < b[1] - res or y > b[3] + res:
                    continue
                row = iy * nx
                for ix in range(ix0, ix1 + 1):
                    x = self.minx + (ix + 0.5) * res
                    if not p.contains_xy(x, y):
                        continue
                    z = w.height(x, y)
                    if p.top <= z + step_up:
                        continue
                    if p.base >= z + mech_h:
                        continue
                    blocked[row + ix] = 1
                    solid[row + ix] = 1
        # Terrain: 1 m samples around each cell centre; >50 deg is unwalkable.
        for iy in range(ny):
            y = self.miny + (iy + 0.5) * res
            row = iy * nx
            for ix in range(nx):
                if blocked[row + ix]:
                    continue
                x = self.minx + (ix + 0.5) * res
                h = w.height(x, y)
                if (w.height(x + 1.0, y) - h > 1.2 or w.height(x - 1.0, y) - h > 1.2
                        or w.height(x, y + 1.0) - h > 1.2 or w.height(x, y - 1.0) - h > 1.2
                        or h - w.height(x + 1.0, y) > 1.2 or h - w.height(x - 1.0, y) > 1.2
                        or h - w.height(x, y + 1.0) > 1.2 or h - w.height(x, y - 1.0) > 1.2):
                    blocked[row + ix] = 1
        # Map edge margin (the map edge is a wall).
        margin = int(30.0 / res)
        for iy in range(ny):
            if iy < margin or iy >= ny - margin:
                row = iy * nx
                for ix in range(nx):
                    blocked[row + ix] = 1
        for ix in range(nx):
            if ix < margin or ix >= nx - margin:
                for iy in range(ny):
                    blocked[iy * nx + ix] = 1
        # Inflate solids by one cell.  The mech radius is 2.5 m; one cell at 5-8 m is conservative
        # but keeps paths well clear of walls, and campus streets are much wider.
        solid = bytearray(blocked)
        for iy in range(1, ny - 1):
            row = iy * nx
            for ix in range(1, nx - 1):
                k = row + ix
                if solid[k]:
                    continue
                if (solid[k - 1] or solid[k + 1] or solid[k - nx] or solid[k + nx]
                        or solid[k - nx - 1] or solid[k - nx + 1] or solid[k + nx - 1] or solid[k + nx + 1]):
                    blocked[k] = 1

    def is_free(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny and not self.blocked[iy * self.nx + ix]

    def is_free_xy(self, x: float, y: float) -> bool:
        ix, iy = self.cell_of(x, y)
        return self.is_free(ix, iy)

    def nearest_free(self, x: float, y: float, max_r: float = 24.0) -> Optional[Vec2]:
        ix, iy = self.cell_of(x, y)
        if self.is_free(ix, iy):
            return self._center(ix, iy)
        max_cells = int(max_r / self.res) + 1
        for r in range(1, max_cells + 1):
            best = None
            bestd = 1e18
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    jx, jy = ix + dx, iy + dy
                    if not self.is_free(jx, jy):
                        continue
                    cx, cy = self._center(jx, jy)
                    d = (cx - x) ** 2 + (cy - y) ** 2
                    if d < bestd:
                        bestd = d
                        best = (cx, cy)
            if best is not None:
                return best
        return None

    def line_free(self, a: Vec2, b: Vec2, step: float = 2.5) -> bool:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        d = math.hypot(dx, dy)
        n = max(1, int(d / step))
        for i in range(n + 1):
            t = i / n
            if not self.is_free_xy(a[0] + dx * t, a[1] + dy * t):
                return False
        return True

    def find_path(self, start: Vec2, goal: Vec2, max_expand: int = 22000,
                  weight: float = 1.25, budget_s: float = 0.012) -> List[Vec2]:
        deadline = time.perf_counter() + budget_s
        sx, sy = start
        gx, gy = goal
        s = self.nearest_free(sx, sy, 10.0)
        g = self.nearest_free(gx, gy, 14.0)
        if s is None or g is None:
            return []
        six, siy = self.cell_of(*s)
        gix, giy = self.cell_of(*g)
        start_k = siy * self.nx + six
        goal_k = giy * self.nx + gix
        if start_k == goal_k:
            return [g]
        nx = self.nx
        blocked = self.blocked
        dirs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.41421356), (1, -1, 1.41421356), (-1, 1, 1.41421356), (-1, -1, 1.41421356))

        def heur(ix: int, iy: int) -> float:
            dx = abs(ix - gix)
            dy = abs(iy - giy)
            return max(dx, dy) + 0.41421356 * min(dx, dy)

        heap = [(heur(six, siy) * weight, 0.0, start_k)]
        gscore = {start_k: 0.0}
        parent: Dict[int, int] = {}
        closed = set()
        expanded = 0
        best_k = start_k
        best_h = heur(six, siy)
        while heap:
            f, gc, k = heappop(heap)
            if k in closed:
                continue
            closed.add(k)
            expanded += 1
            if k == goal_k:
                best_k = k
                break
            ix = k % nx
            iy = k // nx
            h = heur(ix, iy)
            if h < best_h:
                best_h = h
                best_k = k
            if expanded >= max_expand:
                break
            if not (expanded & 255) and time.perf_counter() > deadline:
                break
            for dx, dy, cost in dirs:
                jx = ix + dx
                jy = iy + dy
                if jx < 0 or jy < 0 or jx >= nx or jy >= self.ny:
                    continue
                jk = jy * nx + jx
                if blocked[jk] or jk in closed:
                    continue
                if dx and dy and (blocked[iy * nx + jx] or blocked[jy * nx + ix]):
                    continue
                ng = gc + cost
                old = gscore.get(jk)
                if old is not None and ng >= old:
                    continue
                gscore[jk] = ng
                parent[jk] = k
                heappush(heap, (ng + weight * heur(jx, jy), ng, jk))
        cells: List[int] = []
        k = best_k
        while k != start_k:
            cells.append(k)
            k = parent.get(k)
            if k is None:
                return []
        cells.reverse()
        pts = [self._center(k % nx, k // nx) for k in cells]
        if not pts:
            return [g]
        if best_k != goal_k:
            pts.append(g)
        # Smooth: greedily skip points that are in a clear straight line from the anchor.
        out: List[Vec2] = []
        anchor = s
        for p in pts:
            if self.line_free(anchor, p):
                continue
            if out and out[-1] != anchor:
                anchor = out[-1]
            else:
                anchor = p
            out.append(p)
        if not out or out[-1] != g:
            out.append(g)
        if out and math.hypot(out[0][0] - sx, out[0][1] - sy) < 1.0:
            out.pop(0)
        return out


class World:
    def __init__(self, map_dir: str):
        self.map_dir = map_dir
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        self.raw = w
        self.bounds = w["bounds"]
        t = w["terrain"]
        self.tminx = float(t["minX"])
        self.tminy = float(t["minY"])
        self.tstep = float(t["step"])
        self.tnx = int(t["nx"])
        self.tny = int(t["ny"])
        with open(os.path.join(map_dir, "v2", t["file"]), "rb") as f:
            raw = f.read()
        self.tgrid = array("h")
        self.tgrid.frombytes(raw)
        self.tscale = float(t.get("scale", 0.01))
        self.prisms = [Prism(p) for p in w.get("prisms", [])]
        self.rules = w.get("rules", {})
        # Spatial hash of prisms for ray tests (40 m cells).
        self._hash_cell = 40.0
        self._hash: Dict[Tuple[int, int], List[int]] = {}
        for i, p in enumerate(self.prisms):
            b = p.bbox
            x0 = int(math.floor(b[0] / self._hash_cell))
            x1 = int(math.floor(b[2] / self._hash_cell))
            y0 = int(math.floor(b[1] / self._hash_cell))
            y1 = int(math.floor(b[3] / self._hash_cell))
            for hy in range(y0, y1 + 1):
                for hx in range(x0, x1 + 1):
                    self._hash.setdefault((hx, hy), []).append(i)
        self.nav = NavGrid(self, 8.0)
        self.fine = NavGrid(self, 5.0)

    # ------------------------------------------------------------------ height
    def height(self, x: float, y: float) -> float:
        return _bilinear(
            self.tgrid, self.tnx, self.tny,
            (x - self.tminx) / self.tstep, (y - self.tminy) / self.tstep,
        ) * self.tscale

    def ground_z(self, x: float, y: float) -> float:
        return self.height(x, y)

    # ------------------------------------------------------------------ solids / rays
    def prism_at(self, x: float, y: float, z: float) -> Optional[Prism]:
        for i in self._hash.get((int(math.floor(x / self._hash_cell)), int(math.floor(y / self._hash_cell))), ()):
            p = self.prisms[i]
            if p.base <= z <= p.top and p.contains_xy(x, y):
                return p
        return None

    def prisms_near(self, x: float, y: float, radius: float) -> List[Prism]:
        """Prisms whose bounding box may touch the disc (x, y, radius)."""
        out = []
        seen = set()
        hc = self._hash_cell
        x0 = int(math.floor((x - radius) / hc))
        x1 = int(math.floor((x + radius) / hc))
        y0 = int(math.floor((y - radius) / hc))
        y1 = int(math.floor((y + radius) / hc))
        for hy in range(y0, y1 + 1):
            for hx in range(x0, x1 + 1):
                for i in self._hash.get((hx, hy), ()):
                    if i in seen:
                        continue
                    seen.add(i)
                    p = self.prisms[i]
                    b = p.bbox
                    if b[0] > x + radius or b[2] < x - radius or b[1] > y + radius or b[3] < y - radius:
                        continue
                    out.append(p)
        return out

    def support_height(self, x: float, y: float, z: float, step_up: float = 1.2) -> float:
        """Highest surface at or below `z + step_up` under the point (terrain or roof top)."""
        h = self.height(x, y)
        for p in self.prisms_near(x, y, 0.1):
            if p.top <= z + step_up + 0.02 and p.top > h and p.contains_xy(x, y):
                h = p.top
        return h

    def cylinder_blocked(self, x: float, y: float, z: float, radius: float = 2.5,
                         height: float = 9.0, step_up: float = 1.2, grounded: bool = True,
                         samples: int = 8) -> bool:
        """Approximate a mech cylinder against the prisms.  Terrain is handled by the caller."""
        pts = [(x, y)]
        for i in range(samples):
            a = i * (2.0 * math.pi / samples)
            pts.append((x + math.cos(a) * radius, y + math.sin(a) * radius))
        for px, py in pts:
            for p in self.prisms_near(px, py, 0.01):
                if p.top <= z + 0.02:
                    continue
                if grounded and p.top <= z + step_up + 0.02:
                    continue
                if p.base >= z + height:
                    continue
                if p.contains_xy(px, py):
                    return True
        return False

    def terrain_blocks(self, a: Vec3, b: Vec3, step: float = 1.5) -> bool:
        """Coarse terrain occlusion test for sight and shots."""
        ax, ay, az = a
        bx, by, bz = b
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        d = math.hypot(dx, dy)
        n = max(1, int(d / step))
        for i in range(1, n):
            t = i / n
            x = ax + dx * t
            y = ay + dy * t
            z = az + dz * t
            if self.height(x, y) > z:
                return True
        return False

    def _candidates(self, a: Vec3, b: Vec3) -> List[int]:
        """Prism indices in the hash cells touched by the XY segment."""
        out = set()
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        d = math.hypot(dx, dy)
        n = max(1, int(d / (self._hash_cell * 0.5)))
        for i in range(n + 1):
            t = i / n
            x = a[0] + dx * t
            y = a[1] + dy * t
            hx = int(math.floor(x / self._hash_cell))
            hy = int(math.floor(y / self._hash_cell))
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    v = self._hash.get((hx + ox, hy + oy))
                    if v:
                        out.update(v)
        return list(out)

    def _segment_hits_prism(self, a: Vec3, b: Vec3, p: Prism) -> bool:
        ax, ay, az = a
        bx, by, bz = b
        if max(ax, bx) < p.bbox[0] or min(ax, bx) > p.bbox[2]:
            return False
        if max(ay, by) < p.bbox[1] or min(ay, by) > p.bbox[3]:
            return False
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = max(2, int(length / 1.0) + 1)
        for i in range(n + 1):
            t = i / n
            x = ax + dx * t
            y = ay + dy * t
            z = az + dz * t
            if p.base <= z <= p.top and p.contains_xy(x, y):
                return True
        return False

    def segment_blocked(self, a: Vec3, b: Vec3) -> bool:
        for i in self._candidates(a, b):
            if self._segment_hits_prism(a, b, self.prisms[i]):
                return True
        return False

    def line_of_sight(self, a: Vec3, b: Vec3) -> bool:
        return not self.segment_blocked(a, b)

    def fast_los(self, a: Vec3, b: Vec3) -> bool:
        """Grid LOS used for cover search; exact enough and much cheaper."""
        return self.nav.line_free((a[0], a[1]), (b[0], b[1]), step=3.0)

    def cover_point(self, me: Vec3, threat: Vec3, radius: float = 18.0,
                    zone_center: Vec2 = (0.0, 0.0), zone_radius: float = 1e9) -> Optional[Vec2]:
        ex, ey, ez = threat[0], threat[1], threat[2] + 8.0
        mx, my = me[0], me[1]
        candidates = []
        for i in range(16):
            ang = i * (2.0 * math.pi / 16.0)
            for rr in (radius, radius * 0.65, radius * 0.4):
                x = mx + math.cos(ang) * rr
                y = my + math.sin(ang) * rr
                if math.hypot(x - zone_center[0], y - zone_center[1]) > zone_radius - 3:
                    continue
                if not self.nav.is_free_xy(x, y):
                    continue
                z = self.height(x, y)
                if not self.fast_los((ex, ey, ez), (x, y, z + 4.5)):
                    d = (x - mx) ** 2 + (y - my) ** 2
                    candidates.append((d, x, y, z))
        candidates.sort()
        for d, x, y, z in candidates[:6]:
            if self.segment_blocked((ex, ey, ez), (x, y, z + 4.5)):
                return (x, y)
        return None

    def cover_point_multi(self, me: Vec3, threats: Sequence[Vec3], zone_center: Vec2 = (0.0, 0.0),
                          zone_radius: float = 1e9) -> Optional[Vec2]:
        """Nearby walkable point seen by as few of `threats` as possible (ties: closest, away from them)."""
        if not threats:
            return None
        mx, my = me[0], me[1]
        eyes = [(t[0], t[1], t[2] + 8.0) for t in threats[:4]]
        cx = sum(e[0] for e in eyes) / len(eyes)
        cy = sum(e[1] for e in eyes) / len(eyes)
        away = math.atan2(my - cy, mx - cx)
        best = None
        best_cost = 1e18
        here = sum(1 for e in eyes if self.fast_los(e, (mx, my, 0.0)))
        for i in range(16):
            ang = i * (2.0 * math.pi / 16.0)
            da = abs(math.atan2(math.sin(ang - away), math.cos(ang - away)))
            for rr in (10.0, 20.0, 32.0):
                x = mx + math.cos(ang) * rr
                y = my + math.sin(ang) * rr
                if math.hypot(x - zone_center[0], y - zone_center[1]) > zone_radius - 4:
                    continue
                if not self.nav.is_free_xy(x, y):
                    continue
                if not self.nav.line_free((mx, my), (x, y), step=2.5):
                    continue
                exposed = sum(1 for e in eyes if self.fast_los(e, (x, y, 0.0)))
                cost = exposed * 100.0 + rr * 0.8 + da * 12.0
                if cost < best_cost:
                    best_cost = cost
                    best = (x, y, exposed)
        if best is None or best[2] >= here:
            return None
        return (best[0], best[1])
