"""Mech Royale bot: survival-first positioning, zone play, threat avoidance and opportunistic combat.

The bot reads the full rule set from `hello.config` and the static map from `hello.map.dir`.
It never assumes a particular spawn, zone centre or weapon loadout.
"""
from __future__ import annotations

import math
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import mechfight as mf
from world import World

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

TAU = 2.0 * math.pi


# --------------------------------------------------------------------------- small helpers
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def norm_angle(a: float) -> float:
    while a > math.pi:
        a -= TAU
    while a < -math.pi:
        a += TAU
    return a


def angle_diff(a: float, b: float) -> float:
    return norm_angle(a - b)


def dist2(a: Vec2, b: Vec2) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def dist(a: Vec2, b: Vec2) -> float:
    return math.sqrt(dist2(a, b))


def unit(dx: float, dy: float) -> Vec2:
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return (0.0, 0.0)
    return (dx / d, dy / d)


def yaw_to(a: Vec2, b: Vec2) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def bearing_vec(yaw: float, length: float = 1.0) -> Vec2:
    return (math.cos(yaw) * length, math.sin(yaw) * length)


class Enemy:
    __slots__ = ("id", "kind", "pos", "vel", "hp", "max_hp", "cubes", "weapons", "aim", "charging", "last_t", "seen_t")

    def __init__(self, m: dict, t: float):
        self.id = int(m["id"])
        self.kind = m.get("kind", "mech")
        self.pos = tuple(m["pos"])
        self.vel = tuple(m.get("vel", (0.0, 0.0, 0.0)))
        self.hp = float(m.get("hp", 1400.0))
        self.max_hp = float(m.get("max_hp", 1400.0))
        self.cubes = float(m.get("cubes", 0.0))
        self.weapons = tuple(m.get("weapons") or ())
        self.aim = tuple(m.get("aim", (0.0, 0.0)))
        self.charging = bool(m.get("charging", False))
        self.last_t = t
        self.seen_t = t

    def update(self, m: dict, t: float) -> None:
        self.pos = tuple(m["pos"])
        self.vel = tuple(m.get("vel", (0.0, 0.0, 0.0)))
        self.hp = float(m.get("hp", self.hp))
        self.max_hp = float(m.get("max_hp", self.max_hp))
        self.cubes = float(m.get("cubes", self.cubes))
        self.weapons = tuple(m.get("weapons") or ())
        self.aim = tuple(m.get("aim", self.aim))
        self.charging = bool(m.get("charging", False))
        self.last_t = t
        self.seen_t = t

    @property
    def scale(self) -> float:
        return 1.0 + 0.05 * self.cubes

    def center(self) -> Vec3:
        if self.kind == "npc":
            return (self.pos[0], self.pos[1], self.pos[2] + 1.75)
        return (self.pos[0], self.pos[1], self.pos[2] + 4.5 * self.scale)


class Bot:
    name = "survivor"
    FIGHT_RANGE = 65.0
    FIGHT_RANGE_LIGHTNING = 90.0
    FLEE_HP = 0.42
    HOLD_FIRE_FLEE = False
    FINISH_HP = 260.0
    GUN_TOL_M = 2.4
    CUBE_GREED = 45.0
    MULTI_COVER = True

    def __init__(self, hello: mf.Hello):
        self.cfg = hello["config"]
        self.slot = int(hello["slot"])
        self.rng = random.Random(int(hello["seed"]) & 0xFFFFFFFF)
        self.world = World(hello["map"]["dir"])

        m = self.cfg["mech"]
        self.radius = float(m["radius"])
        self.height = float(m["height"])
        self.eye_h = float(m["eye_height"])
        self.muzzle_h = float(m["muzzle_height"])
        self.walk_speed = float(m["walk_speed"])
        self.boost_speed = float(m["boost_speed"])
        self.turn_rate = float(m["turn_rate"])
        self.pitch_rate = float(m["pitch_rate"])
        self.min_pitch = float(m["min_pitch"])
        self.max_pitch = float(m["max_pitch"])
        self.step_up = float(m["step_up"])
        self.max_energy = float(m["energy"])
        self.energy_regen = float(m["energy_regen"])
        self.boost_drain = float(m["boost_drain"])
        self.jet_drain = float(m["jet_drain"])
        self.jump_speed = float(m["jump_speed"])
        self.gravity = float(m["gravity"])

        self.vision_range = float(self.cfg["vision"]["range"])
        self.near_range = float(self.cfg["vision"]["near_range"])
        self.fov = math.radians(float(self.cfg["vision"]["fov_deg"]))
        self.memory_s = float(self.cfg["vision"].get("memory_s", 5.0))

        self.gun = dict(self.cfg["weapons"]["gun"])
        self.mortar = dict(self.cfg["weapons"]["mortar"])
        self.laser = dict(self.cfg["weapons"]["laser"])
        self.lightning = dict(self.cfg["weapons"]["lightning"])
        self.loot = dict(self.cfg["loot"])
        self.cubes_cfg = dict(self.cfg["cubes"])
        self.crates_cfg = dict(self.cfg["crates"])
        self.npc_cfg = dict(self.cfg["npc"])
        self.drops_cfg = dict(self.cfg.get("drops") or {})
        self.zone_cfg = dict(self.cfg["zone"])

        self.t = 0.0
        self.tick = 0
        self.last_obs: Optional[dict] = None
        self.self_pos: Vec3 = (0.0, 0.0, 0.0)
        self.self_vel: Vec3 = (0.0, 0.0, 0.0)
        self.self_aim: Vec2 = (0.0, 0.0)
        self.hp = float(m["hp"])
        self.max_hp = float(m["hp"])
        self.energy = self.max_energy
        self.grounded = True
        self.cubes = 0.0
        self.weapons: List[Optional[dict]] = [None, None]
        self.in_zone = True
        self.upgrades: List[str] = []
        self.stats = dict(m)

        self.enemies: Dict[int, Enemy] = {}
        self.sound_threats: List[Tuple[float, Vec2, str, float]] = []  # t, pos, kind, confidence
        self.last_damage_t = -100.0
        self.last_attacker: Optional[int] = None
        self.last_damage_bearing: Optional[float] = None
        self.last_seen_enemy_t = -100.0
        self.last_shot_t = -100.0
        self.last_hit_t = -100.0
        self.last_kill_t = -100.0

        self.goal: Optional[Tuple[float, float, str, Optional[int]]] = None
        self.goal_since = 0.0
        self.path: List[Vec2] = []
        self.path_i = 0
        self.path_goal: Optional[Vec2] = None
        self.path_t = -100.0
        self.path_mode = ""
        self.stuck_count = 0
        self.last_move_pos: Vec3 = (0.0, 0.0, 0.0)
        self.last_move_t = -100.0
        self.last_move_cmd: Vec2 = (0.0, 0.0)
        self.cover_target: Optional[Vec2] = None
        self.cover_t = -100.0
        self.cover_threat: Optional[Vec3] = None

        self.mode = "rotate"
        self.combat_target: Optional[int] = None
        self.combat_since = 0.0
        self.farm_target: Optional[Tuple[str, int]] = None  # ("crate"/"npc", id)
        self.farm_since = 0.0
        self.laser_charging = False
        self.last_offer_id: Optional[int] = None
        self.pending_take: Optional[int] = None
        self.pending_take_t = -100.0
        self.last_reload_t = -100.0
        self.last_jump_t = -100.0
        self.last_boost_t = -100.0
        self.scan_phase = 0.0
        self.final_goal: Optional[Vec2] = None
        self.climb: Optional[dict] = None
        self.climb_since = 0.0
        self.start_t = 0.0
        self.debug = False

    # ------------------------------------------------------------------ main
    def act(self, obs: mf.Observation) -> mf.Action:
        now = time.perf_counter()
        self.last_obs = obs
        self.t = float(obs["t"])
        self.tick = int(obs["tick"])
        me = obs["self"]
        self.self_pos = tuple(me["pos"])
        self.self_vel = tuple(me["vel"])
        self.self_aim = tuple(me["aim"])
        self.hp = float(me["hp"])
        self.max_hp = float(me["max_hp"])
        self.energy = float(me["energy"])
        self.grounded = bool(me["grounded"])
        self.cubes = float(me["cubes"])
        self.weapons = list(me["weapons"])
        self.in_zone = bool(me["in_zone"])
        self.upgrades = list(me.get("upgrades") or ())
        self.stats = dict(me.get("stats") or {})
        self._apply_stats()

        self._update_enemies(obs)
        self._update_sounds(obs)
        self._process_events(obs)

        action: mf.Action = {"tick": obs["tick"]}

        # Upgrade offers are time-limited: answer immediately.
        offer = obs.get("offer")
        if offer:
            action["pick"] = self._choose_upgrade(offer)

        # Determine the strategic mode and goal.
        mode, goal = self._choose_mode(obs)
        self.mode = mode
        if goal is not None:
            if self.goal is None or goal[2] != self.goal[2] or goal[3] != self.goal[3] or dist((goal[0], goal[1]), (self.goal[0], self.goal[1])) > 18:
                self.goal = goal
                self.goal_since = self.t
        self.goal = goal

        # Movement.
        move, boost, jump = self._movement(obs, mode, goal, now)

        # Aim and fire.
        aim, fire = self._combat(obs, mode)
        if mode == "flee" and self.HOLD_FIRE_FLEE and self.t - self.last_damage_t > 1.0:
            tgt = self.enemies.get(self.combat_target) if self.combat_target is not None else None
            if tgt is None or tgt.hp > self.FINISH_HP:
                # Dealing damage resets the regen timer: stay quiet and heal.
                fire = (False, fire[1] and self.laser_charging)
        if aim is None:
            aim = self._scan_aim(obs, move)
        action["move"] = [float(move[0]), float(move[1])]
        action["boost"] = bool(boost)
        action["jump"] = bool(jump)
        action["aim"] = [float(aim[0]), float(aim[1])]
        action["fire"] = [bool(fire[0]), bool(fire[1])]

        # Weapon pickup / reload.
        take, drop = self._take_decision(obs)
        if take is not None:
            action["take"] = int(take)
            if drop is not None:
                action["drop"] = int(drop)
        if self._should_reload(obs):
            action["reload"] = True

        if self.debug:
            self._draw(obs, aim, mode)
        return action

    # ------------------------------------------------------------------ state updates
    def _apply_stats(self) -> None:
        s = self.stats
        self.walk_speed = float(s.get("walk_speed", self.walk_speed))
        self.boost_speed = float(s.get("boost_speed", self.boost_speed))
        self.turn_rate = float(s.get("turn_rate", self.turn_rate))
        self.pitch_rate = float(s.get("pitch_rate", self.pitch_rate))
        self.max_energy = float(s.get("energy", self.max_energy))
        self.energy_regen = float(s.get("energy_regen", self.energy_regen))
        self.boost_drain = float(s.get("boost_drain", self.boost_drain))
        self.jet_drain = float(s.get("jet_drain", self.jet_drain))
        self.eye_h = float(s.get("eye_height", self.eye_h))
        self.muzzle_h = float(s.get("muzzle_height", self.muzzle_h))
        self.radius = float(s.get("radius", self.radius))
        self.height = float(s.get("height", self.height))
        self.vision_range = float(self.cfg["vision"]["range"]) + float(s.get("vision_bonus", 0.0))
        self.near_range = float(self.cfg["vision"]["near_range"]) + float(s.get("near_range_bonus", 0.0))

    def _update_enemies(self, obs: dict) -> None:
        t = self.t
        seen = set()
        for m in obs.get("mechs") or ():
            mid = int(m["id"])
            seen.add(mid)
            e = self.enemies.get(mid)
            if e is None:
                self.enemies[mid] = Enemy(m, t)
            else:
                e.update(m, t)
            if m.get("kind") == "mech":
                self.last_seen_enemy_t = t
        # forget stale memories
        for mid in list(self.enemies):
            e = self.enemies[mid]
            if mid not in seen and t - e.last_t > 8.0:
                del self.enemies[mid]

    def _update_sounds(self, obs: dict) -> None:
        t = self.t
        near = float(self.cfg["sound"].get("near", 60.0))
        mid = float(self.cfg["sound"].get("mid", 160.0))
        for s in obs.get("sounds") or ():
            kind = s.get("kind", "")
            if kind not in ("gun", "mortar", "explosion", "laser", "laser_charge", "lightning", "npc_gun", "death", "drop", "crate", "boost", "jet"):
                continue
            d = s.get("dist", "far")
            if d == "near":
                dd = near * 0.5
                conf = 0.7
            elif d == "mid":
                dd = (near + mid) * 0.5
                conf = 0.4
            else:
                dd = 230.0
                conf = 0.15
            age = float(s.get("age", 0.0))
            bearing = float(s.get("bearing", 0.0))
            pos = (self.self_pos[0] + math.cos(bearing) * dd,
                   self.self_pos[1] + math.sin(bearing) * dd)
            self.sound_threats.append((t - age, pos, kind, conf))
        # keep the last 6 seconds
        self.sound_threats = [x for x in self.sound_threats if t - x[0] < 6.0][-40:]

    def _process_events(self, obs: dict) -> None:
        for e in obs.get("events") or ():
            typ = e.get("type")
            if typ == "damaged":
                self.last_damage_t = self.t
                by = e.get("by")
                self.last_attacker = int(by) if by is not None else None
                self.last_damage_bearing = e.get("bearing")
                if self.last_damage_bearing is not None:
                    # Face the attacker next tick; this also reduces rear-armour hits.
                    self.scan_phase = 0.0
            elif typ == "hit":
                self.last_hit_t = self.t
                tid = e.get("target")
                if tid is not None:
                    en = self.enemies.get(int(tid))
                    if en is not None and e.get("weapon") in ("gun", "laser", "lightning", "mortar"):
                        pass
            elif typ == "kill":
                self.last_kill_t = self.t
            elif typ == "pickup":
                self.cubes = float(e.get("cubes", self.cubes))
            elif typ == "drop":
                self.cubes = float(e.get("cubes", self.cubes))
            elif typ == "took":
                self.pending_take = None
            elif typ == "take_failed":
                self.pending_take = None
            elif typ == "effect":
                # Cook-off / short / concuss: back off and let the effect expire.
                if e.get("effect") in ("short", "cookoff"):
                    self.laser_charging = False

    # ------------------------------------------------------------------ strategy
    def _zone_info(self, obs: dict) -> Tuple[Vec2, float, Vec2, float, float, float, float]:
        z = obs["zone"]
        cx, cy = z["center"]
        nx, ny = z["next_center"]
        return ((cx, cy), float(z["radius"]), (nx, ny), float(z["next_radius"]),
                float(z["shrink_start"]), float(z["shrink_end"]), float(z["dps"]))

    def _zone_goal(self, obs: dict) -> Tuple[Vec2, str]:
        center, radius, ncenter, nradius, ss, se, dps = self._zone_info(obs)
        x, y = self.self_pos[0], self.self_pos[1]
        t = self.t
        dcur = dist((x, y), center)
        # Final circle: go to the exact next centre as soon as it is revealed.
        if nradius <= 1.0:
            self.final_goal = ncenter
            return ncenter, "final"
        if radius <= 25.0:
            self.final_goal = center
            return center, "final"
        # If outside (or nearly outside) the current circle, run to its centre.
        if dcur > radius - 8.0:
            return center, "zone_in"
        # If shrinking or about to shrink, get inside the next circle.
        if nradius < radius and t >= ss - 7.0:
            dn = dist((x, y), ncenter)
            if dn > nradius - 15.0:
                return ncenter, "zone_next"
            # already comfortably inside the next circle: hold near its centre
            return ncenter, "zone_hold"
        # Early game: stay near the current centre but with a little offset so we are not a fixed target.
        if radius > 80.0:
            # choose a point inside the current circle, biased toward the next centre
            ang = (self.slot * 1.7 + t * 0.02) % TAU
            off = min(radius * 0.45, 70.0)
            px = center[0] + math.cos(ang) * off
            py = center[1] + math.sin(ang) * off
            if dist((px, py), center) < radius - 20.0:
                return (px, py), "patrol"
        return center, "zone_hold"

    def _threat_summary(self) -> Tuple[Optional[Enemy], float, int, int]:
        """Nearest visible mech, its distance, count within 120 m, count within 60 m."""
        best = None
        bestd = 1e18
        n120 = 0
        n60 = 0
        x, y = self.self_pos[0], self.self_pos[1]
        for e in self.enemies.values():
            if e.kind != "mech":
                continue
            if self.t - e.last_t > self.memory_s + 1.0:
                continue
            d = math.hypot(e.pos[0] - x, e.pos[1] - y)
            if d < 120:
                n120 += 1
            if d < 60:
                n60 += 1
            if d < bestd:
                bestd = d
                best = e
        return best, (bestd if best is not None else 1e18), n120, n60

    def _choose_mode(self, obs: dict) -> Tuple[str, Optional[Tuple[float, float, str, Optional[int]]]]:
        x, y = self.self_pos[0], self.self_pos[1]
        hp_frac = self.hp / max(1.0, self.max_hp)
        enemy, edist, n120, n60 = self._threat_summary()
        zone_goal, zkind = self._zone_goal(obs)
        zx, zy = zone_goal
        zone_dist = math.hypot(zx - x, zy - y)
        outside = not self.in_zone

        # 1. Zone always wins when it is dangerous to stay.
        if outside or zkind in ("zone_in",) or (zkind in ("zone_next", "final") and zone_dist > 8.0 and self.t > 40):
            # If a single enemy is right on top of us, still shoot while moving.
            return "rotate", (zx, zy, "zone", None)

        # 2. Final phase: hold the final centre.
        if zkind == "final":
            return "final", (zx, zy, "final", None)

        # 3. Low health: disengage and heal.
        if hp_frac < self.FLEE_HP and edist < 190.0:
            cover = self._find_cover(obs, enemy)
            if cover is not None:
                return "flee", (cover[0], cover[1], "cover", None)
            if enemy is not None:
                # run away from the enemy, but stay in the zone
                dx = x - enemy.pos[0]
                dy = y - enemy.pos[1]
                ux, uy = unit(dx, dy)
                gx = clamp(x + ux * 70.0, self.world.bounds["minX"] + 40, self.world.bounds["maxX"] - 40)
                gy = clamp(y + uy * 70.0, self.world.bounds["minY"] + 40, self.world.bounds["maxY"] - 40)
                if math.hypot(gx - zx, gy - zy) > max(30.0, self._zone_info(obs)[1] - 10.0):
                    gx, gy = zx, zy
                return "flee", (gx, gy, "flee", None)
            return "rotate", (zx, zy, "zone", None)

        # 4. Multiple hunters close: break line of sight and run.
        if n120 >= 3 and edist < 150.0:
            cover = self._find_cover(obs, enemy)
            if cover is not None:
                return "flee", (cover[0], cover[1], "cover", None)
            return "rotate", (zx, zy, "zone", None)

        # 4b. Cubes lying close by (usually from a kill) are worth grabbing even in a fight.
        if self.CUBE_GREED > 0 and self.cubes < 10 and hp_frac > 0.3:
            center, radius = self._zone_info(obs)[:2]
            bc = None
            bd = self.CUBE_GREED
            for c in obs.get("cubes") or ():
                cx_, cy_ = float(c["pos"][0]), float(c["pos"][1])
                dd = math.hypot(cx_ - x, cy_ - y)
                if dd < bd and math.hypot(cx_ - center[0], cy_ - center[1]) < radius - 5.0:
                    bd = dd
                    bc = (cx_, cy_, "cube", int(c["id"]))
            if bc is not None:
                return "loot", bc

        # 5. A single enemy in comfortable range: fight only if it is worth it, otherwise keep moving.
        if enemy is not None and edist < 170.0:
            e_hp = enemy.hp / max(1.0, enemy.max_hp)
            advantage = hp_frac > e_hp + 0.12
            dangerous = "lightning" in enemy.weapons and edist < 55.0
            if edist < 55.0 and (advantage or e_hp < 0.3 or "lightning" in self._secondary_kinds()):
                return "fight", (x, y, "fight", enemy.id)
            if edist < 100.0 and advantage and e_hp < 0.55:
                return "fight", (x, y, "fight", enemy.id)
            if edist < 35.0:
                # Too close to ignore; fight or flee depending on health.
                return ("fight" if hp_frac > 0.55 and not dangerous else "flee"), (x, y, "fight", enemy.id)
            # Otherwise use cover and keep rotating.
            if edist < 120.0:
                cover = self._find_cover(obs, enemy)
                if cover is not None:
                    return "rotate", (cover[0], cover[1], "cover", None)

        # 6. Loot / farm when the area is reasonably safe.
        loot_goal = self._choose_loot(obs, edist, n120)
        if loot_goal is not None:
            return "loot", loot_goal

        # 7. Default: zone play.
        return "rotate", (zx, zy, "zone", None)

    def _secondary_kinds(self) -> Tuple[str, ...]:
        w = self.weapons[1] if len(self.weapons) > 1 else None
        return (w.get("kind"),) if w else ()

    def _choose_loot(self, obs: dict, edist: float, n120: int) -> Optional[Tuple[float, float, str, Optional[int]]]:
        x, y = self.self_pos[0], self.self_pos[1]
        t = self.t
        if self.cubes >= 10.0:
            cube_value = 0.0
        else:
            cube_value = 1.0
        # Do not loot when hunters are close.
        if edist < 110.0 or n120 >= 2:
            return None
        center, radius, ncenter, nradius, ss, se, dps = self._zone_info(obs)
        if radius < 120.0 and t > 250.0:
            return None

        best = None
        best_score = 0.0

        def consider(px: float, py: float, kind: str, ident: Optional[int], base: float, max_d: float) -> None:
            nonlocal best, best_score
            d = math.hypot(px - x, py - y)
            if d > max_d:
                return
            # stay inside the current zone with a small margin
            if math.hypot(px - center[0], py - center[1]) > radius - 6.0:
                return
            score = base / (d + 25.0)
            if score > best_score:
                best_score = score
                best = (px, py, kind, ident)

        # Supply drops are the best prize, but only when we can get there alone.
        for d in obs.get("drops") or ():
            if float(d.get("lands_in", 1.0)) <= 0.0:
                holder = d.get("holder")
                if holder is None or int(holder) == self.slot:
                    consider(float(d["pos"][0]), float(d["pos"][1]), "drop", int(d["id"]), 180.0, 260.0)

        # Weapon pickups: only if the second slot is empty (or a clearly better weapon is available).
        sec = self.weapons[1] if len(self.weapons) > 1 else None
        for p in obs.get("pickups") or ():
            kind = p.get("kind")
            if kind not in ("laser", "lightning", "mortar"):
                continue
            if not self._want_weapon(kind):
                continue
            pref = {"laser": 170.0, "lightning": 90.0, "mortar": 70.0}.get(kind, 50.0)
            consider(float(p["pos"][0]), float(p["pos"][1]), "pickup", int(p["id"]), pref, 200.0 if kind == "laser" else 140.0)

        # Cubes.
        if cube_value > 0:
            for c in obs.get("cubes") or ():
                consider(float(c["pos"][0]), float(c["pos"][1]), "cube", int(c["id"]), 70.0, 90.0)

        # Crates and NPCs for cubes, only when no mech is near.
        if cube_value > 0 and edist > 160.0:
            for c in obs.get("crates") or ():
                consider(float(c["pos"][0]), float(c["pos"][1]), "crate", int(c["id"]), 45.0, 110.0)
            for m in obs.get("mechs") or ():
                if m.get("kind") == "npc":
                    consider(float(m["pos"][0]), float(m["pos"][1]), "npc", int(m["id"]), 55.0, 100.0)
        return best

    def _find_cover(self, obs: dict, enemy: Optional[Enemy]) -> Optional[Vec2]:
        if enemy is None:
            return None
        if self.cover_target is not None and self.t - self.cover_t < 1.5:
            if self.cover_threat is not None and math.hypot(enemy.pos[0] - self.cover_threat[0], enemy.pos[1] - self.cover_threat[1]) < 25.0:
                return self.cover_target
        center, radius = self._zone_info(obs)[:2]
        x, y = self.self_pos[0], self.self_pos[1]
        threats = sorted((e for e in self.enemies.values()
                          if e.kind == "mech" and self.t - e.last_t < 1.0 and math.hypot(e.pos[0] - x, e.pos[1] - y) < 220.0),
                         key=lambda e: math.hypot(e.pos[0] - x, e.pos[1] - y))
        cp = None
        if self.MULTI_COVER and len(threats) >= 2:
            cp = self.world.cover_point_multi(self.self_pos, [e.pos for e in threats], center, radius)
        if cp is None:
            cp = self.world.cover_point(self.self_pos, enemy.pos, 22.0, center, radius)
        self.cover_t = self.t
        self.cover_threat = enemy.pos
        self.cover_target = cp
        return cp

    # ------------------------------------------------------------------ movement
    def _plan(self, goal: Vec2, force: bool = False) -> None:
        if not force and self.path and self.path_goal is not None:
            if dist(goal, self.path_goal) < 16.0 and self.t - self.path_t < 1.8:
                return
        x, y = self.self_pos[0], self.self_pos[1]
        d = math.hypot(goal[0] - x, goal[1] - y)
        if d < 1.5:
            self.path = []
            self.path_i = 0
            self.path_goal = goal
            self.path_t = self.t
            return
        if d < 90.0:
            p = self.world.fine.find_path((x, y), goal, max_expand=7000)
            if not p:
                p = self.world.nav.find_path((x, y), goal, max_expand=18000)
        else:
            p = self.world.nav.find_path((x, y), goal, max_expand=20000)
            if not p:
                p = self.world.fine.find_path((x, y), goal, max_expand=9000)
        self.path = p
        self.path_i = 0
        self.path_goal = goal
        self.path_t = self.t
        self.path_mode = self.mode

    def _follow_path(self, goal: Vec2) -> Vec2:
        x, y = self.self_pos[0], self.self_pos[1]
        if not self.path:
            return unit(goal[0] - x, goal[1] - y)
        # skip waypoints we have reached or passed
        while self.path_i < len(self.path):
            wx, wy = self.path[self.path_i]
            if math.hypot(wx - x, wy - y) < 5.0:
                self.path_i += 1
                continue
            break
        if self.path_i >= len(self.path):
            return unit(goal[0] - x, goal[1] - y)
        wx, wy = self.path[self.path_i]
        # if the next waypoint is somehow blocked, replan
        if not self.world.nav.is_free_xy(wx, wy):
            self.path = []
            return unit(goal[0] - x, goal[1] - y)
        return unit(wx - x, wy - y)

    def _movement(self, obs: dict, mode: str, goal: Optional[Tuple[float, float, str, Optional[int]]], now: float) -> Tuple[Vec2, bool, bool]:
        x, y, z = self.self_pos
        hp_frac = self.hp / max(1.0, self.max_hp)
        enemy, edist, n120, n60 = self._threat_summary()
        center, radius, ncenter, nradius, ss, se, dps = self._zone_info(obs)
        urgent = (not self.in_zone) or (self.t >= ss - 2.0 and radius > 25.0 and dist((x, y), ncenter) > nradius - 8.0) or mode == "flee"
        silent = (edist < 130.0) and not urgent and mode not in ("fight",)
        if mode == "fight":
            silent = False

        # Default direction toward the goal.
        if goal is not None:
            gx, gy = goal[0], goal[1]
            if mode in ("loot", "final", "rotate", "flee"):
                self._plan((gx, gy))
                d = self._follow_path((gx, gy))
            else:
                d = unit(gx - x, gy - y)
        else:
            d = (0.0, 0.0)

        # Combat positioning: keep range and strafe instead of charging.
        if mode == "fight" and enemy is not None:
            ex, ey = enemy.pos[0], enemy.pos[1]
            ddx, ddy = x - ex, y - ey
            ed = math.hypot(ddx, ddy)
            ux, uy = unit(ddx, ddy)
            desired = self.FIGHT_RANGE
            if "lightning" in enemy.weapons:
                desired = max(self.FIGHT_RANGE, self.FIGHT_RANGE_LIGHTNING)
            radial = 0.0
            if ed < desired - 8.0:
                radial = -1.0  # back away
            elif ed > desired + 18.0:
                radial = 1.0   # close a little
            # strafe perpendicular
            px, py = -uy, ux
            strafe = 1.0 if (self.slot + int(self.t * 0.7)) % 2 == 0 else -1.0
            d = unit(ux * radial + px * strafe * 0.9, uy * radial + py * strafe * 0.9)

        # Flee: run away from the threat, but do not leave the zone.
        if mode == "flee" and enemy is not None and (goal is None or goal[2] == "flee"):
            ax = ay = 0.0
            for e in self.enemies.values():
                if e.kind != "mech" or self.t - e.last_t > 1.0:
                    continue
                dd = math.hypot(x - e.pos[0], y - e.pos[1])
                if dd < 220.0:
                    ux, uy = unit(x - e.pos[0], y - e.pos[1])
                    w = 1.0 / max(20.0, dd)
                    ax += ux * w
                    ay += uy * w
            d = unit(ax, ay) if (ax or ay) else unit(x - enemy.pos[0], y - enemy.pos[1])

        # Dodge incoming projectiles / lasers.
        dodge, dodge_urgent = self._dodge(obs)
        if dodge is not None:
            d = unit(d[0] + dodge[0] * 1.7, d[1] + dodge[1] * 1.7)
            urgent = urgent or dodge_urgent

        # If the goal is very close, stop (except in fight/flee/final).
        if goal is not None and mode in ("loot", "rotate", "zone", "final"):
            gd = math.hypot(goal[0] - x, goal[1] - y)
            if gd < 2.0 and mode != "final":
                d = (0.0, 0.0)
            elif gd < 7.0 and goal[2] in ("pickup", "cube", "drop"):
                d = (d[0] * 0.5, d[1] * 0.5)

        # Speed control.
        speed = 1.0
        if silent and not urgent:
            speed = 0.38
        if hp_frac < 0.3 and mode != "flee":
            speed *= 0.9
        move = (d[0] * speed, d[1] * speed)

        # Stuck detection: if we have been trying to move but haven't, jump and replan.
        if self.t - self.last_move_t > 0.6:
            moved = math.hypot(x - self.last_move_pos[0], y - self.last_move_pos[1])
            if self.last_move_t > 0 and moved < 0.6 and math.hypot(self.last_move_cmd[0], self.last_move_cmd[1]) > 0.3:
                self.stuck_count += 1
            else:
                self.stuck_count = max(0, self.stuck_count - 1)
            self.last_move_pos = (x, y, z)
            self.last_move_t = self.t
            self.last_move_cmd = move

        boost = False
        jump = False
        # Boost on the ground for long rotations, but stay quiet near enemies.
        if self.grounded:
            if self.energy > 12.0 and (urgent or (math.hypot(d[0], d[1]) > 0.5 and not silent and (goal is None or math.hypot(goal[0] - x, goal[1] - y) > 35.0))):
                if not self.laser_charging:
                    boost = True
            if self.stuck_count > 0 and self.energy > 8.0:
                jump = True
                boost = True
        else:
            # In the air: jet only when climbing or stuck.
            if self.energy > 8.0 and (self.stuck_count > 0 or self._need_climb(goal, d)):
                boost = True
        # Jump over small obstacles / when a wall is right ahead.
        if self.grounded and self.energy > 5.0 and self._obstacle_ahead(d):
            jump = True
        # Never boost while charging a laser (it halves movement anyway) unless fleeing.
        if self.laser_charging and not urgent:
            boost = False

        return move, boost, jump

    def _obstacle_ahead(self, d: Vec2) -> bool:
        if math.hypot(d[0], d[1]) < 0.2:
            return False
        x, y, z = self.self_pos
        for step in (3.0, 5.5):
            px = x + d[0] * step
            py = y + d[1] * step
            if not self.world.nav.is_free_xy(px, py):
                h = self.world.height(px, py)
                if h > z + 0.6:
                    return True
                # a wall that is low enough to jump onto
                return True
        return False

    def _need_climb(self, goal: Optional[Tuple[float, float, str, Optional[int]]], d: Vec2) -> bool:
        if goal is None:
            return False
        gx, gy = goal[0], goal[1]
        gz = self.world.height(gx, gy)
        if gz > self.self_pos[2] + 2.0 and math.hypot(gx - self.self_pos[0], gy - self.self_pos[1]) < 30.0:
            return True
        return False

    def _dodge(self, obs: dict) -> Tuple[Optional[Vec2], bool]:
        x, y, z = self.self_pos
        cx, cy, cz = x, y, z + 4.5 * (1.0 + 0.05 * self.cubes)
        vx, vy, vz = self.self_vel
        best = None
        best_w = 0.0
        urgent = False
        for p in obs.get("projectiles") or ():
            kind = p.get("kind")
            if kind not in ("gun", "npc", "mortar"):
                continue
            px, py, pz = p["pos"]
            pvx, pvy, pvz = p["vel"]
            # relative motion in 3D, but dodge horizontally
            rx, ry, rz = px - cx, py - cy, pz - cz
            rvx, rvy = pvx - vx, pvy - vy
            vv = rvx * rvx + rvy * rvy
            if vv < 1e-6:
                continue
            tt = -(rx * rvx + ry * rvy) / vv
            if tt < 0.0 or tt > 1.4:
                continue
            mx = rx + rvx * tt
            my = ry + rvy * tt
            miss = math.hypot(mx, my)
            hit_r = self.radius + (0.9 if kind == "mortar" else 0.5)
            if miss < hit_r + 2.5:
                # perpendicular to the projectile's horizontal velocity
                ux, uy = unit(rvx, rvy)
                pxp, pyp = -uy, ux
                # choose the side that moves us toward the zone centre / cover
                center, radius, ncenter, nradius, ss, se, dps = self._zone_info(obs)
                side = 1.0 if dist((x + pxp * 8, y + pyp * 8), center) < dist((x - pxp * 8, y - pyp * 8), center) else -1.0
                w = 1.0 + max(0.0, 1.0 - tt)
                if w > best_w:
                    best_w = w
                    best = (pxp * side, pyp * side)
                urgent = urgent or (kind == "mortar" and tt < 0.9)
        # Enemy laser charge: do not stand in the beam.
        for e in self.enemies.values():
            if e.kind != "mech" or not e.charging:
                continue
            d = math.hypot(e.pos[0] - x, e.pos[1] - y)
            if d > 300.0:
                continue
            to_us = math.atan2(y - e.pos[1], x - e.pos[0])
            if abs(angle_diff(to_us, e.aim[0])) < 0.22:
                ux, uy = unit(x - e.pos[0], y - e.pos[1])
                pxp, pyp = -uy, ux
                side = 1.0 if (self.slot + int(self.t * 1.3)) % 2 == 0 else -1.0
                if 1.3 > best_w:
                    best_w = 1.3
                    best = (pxp * side, pyp * side)
                urgent = True
        return best, urgent

    # ------------------------------------------------------------------ aiming / firing
    def _combat(self, obs: dict, mode: str) -> Tuple[Optional[Vec2], Tuple[bool, bool]]:
        x, y, z = self.self_pos
        muzzle = (x, y, z + self.muzzle_h)
        # Candidate targets: every mech the server tells us about (memory entries are real positions).
        best = None
        best_score = 1e18
        for e in self.enemies.values():
            if e.kind != "mech" or self.t - e.last_t > 0.35:
                continue
            d = math.hypot(e.pos[0] - x, e.pos[1] - y)
            if d > 280.0:
                continue
            los = self.world.line_of_sight(muzzle, e.center())
            score = d + (e.hp / max(1.0, e.max_hp)) * 90.0 + e.hp * 0.03
            if not los:
                score += 400.0
            if e.id == self.combat_target:
                score -= 35.0
            if "lightning" in e.weapons and d < 55.0:
                score -= 40.0
            if e.charging:
                score -= 30.0
            if score < best_score:
                best_score = score
                best = (e, d, los)
        if best is None:
            self.combat_target = None
            self.laser_charging = False
            return self._shoot_pve(obs, mode)
        target, d, los = best
        self.combat_target = target.id

        sec = self.weapons[1] if len(self.weapons) > 1 else None
        sec_kind = sec.get("kind") if sec else None
        gun_aim = self._aim_point(target, "gun", d)
        gun_range = float(self.gun.get("range", 230.0))
        gun_ok = self._gun_ready() and d <= gun_range * 0.98
        fire0 = False
        fire1 = False
        aim = gun_aim

        if sec_kind == "laser":
            st = sec["stats"]
            charge = float(sec.get("charge", 0.0))
            full = float(st.get("charge", 0.75))
            ready = float(sec.get("cooldown", 0.0)) <= 0.01
            laser_aim = self._aim_point(target, "laser", d)
            if los and d <= float(st.get("range", 280.0)) * 0.97 and (ready or charge > 0.0) and mode != "flee_hard":
                aim = laser_aim
                err = self._aim_error(*laser_aim)
                tol = clamp(math.atan2(2.0, max(10.0, d)), 0.006, 0.08)
                if charge < full * 0.97:
                    fire1 = True  # keep charging
                elif err > tol and charge < full + float(st.get("hold_s", 2.0)) * 0.8:
                    fire1 = True  # hold until on target
                else:
                    fire1 = False  # release: fires now
                self.laser_charging = fire1
            elif charge > 0.0:
                # Lost the shot while charging: keep holding a little so the beam is not wasted.
                aim = laser_aim
                fire1 = charge < full + float(st.get("hold_s", 2.0)) * 0.6
                self.laser_charging = fire1
            else:
                self.laser_charging = False
        elif sec_kind == "lightning":
            self.laser_charging = False
            st = sec["stats"]
            if los and d <= float(st.get("range", 45.0)) * 0.93 and self.energy >= float(st.get("energy", 3.0)) + 1.0 \
                    and float(sec.get("cooldown", 0.0)) <= 0.01:
                yerr = abs(angle_diff(self.self_aim[0], gun_aim[0]))
                if yerr < math.radians(float(st.get("cone_deg", 30.0))) * 0.4:
                    fire1 = True
        elif sec_kind == "mortar":
            self.laser_charging = False
            ready = float(sec.get("cooldown", 0.0)) <= 0.01
            if ready and 32.0 <= d <= 225.0 and (not los or not gun_ok):
                v = float(self.mortar.get("speed", 72.0))
                maim = None
                for high in ((False, True) if los else (True,)):
                    a0 = self._mortar_aim(target.pos[0], target.pos[1], target.pos[2] + 1.0, d, high)
                    if a0 is None:
                        continue
                    tof = d / max(1.0, v * math.cos(a0[1]))
                    lx = target.pos[0] + target.vel[0] * tof * 0.8
                    ly = target.pos[1] + target.vel[1] * tof * 0.8
                    a1 = self._mortar_aim(lx, ly, target.pos[2] + 1.0, math.hypot(lx - x, ly - y), high)
                    if a1 is not None and self._mortar_clear(a1):
                        maim = a1
                        break
                if maim is not None:
                    aim = maim
                    if self._aim_error(*maim) < 0.03:
                        fire1 = True
        else:
            self.laser_charging = False

        if gun_ok and los:
            err = self._aim_error(gun_aim[0], gun_aim[1])
            tol = clamp(math.atan2(self.GUN_TOL_M, max(12.0, d)), 0.008, 0.07)
            if err < tol:
                fire0 = True
        return aim, (fire0, fire1)

    def _shoot_pve(self, obs: dict, mode: str) -> Tuple[Optional[Vec2], Tuple[bool, bool]]:
        """No mech in sight: shoot NPC drones (they shoot back) and the crate we are farming."""
        x, y, z = self.self_pos
        muzzle = (x, y, z + self.muzzle_h)
        tgt = None
        best = 1e18
        for e in self.enemies.values():
            if e.kind != "npc" or self.t - e.last_t > 0.35:
                continue
            d = math.hypot(e.pos[0] - x, e.pos[1] - y)
            if d > 120.0:
                continue
            if not self.world.line_of_sight(muzzle, e.center()):
                continue
            sc = d - (60.0 if self.t - self.last_damage_t < 1.5 else 0.0)
            if sc < best:
                best = sc
                tgt = (e.pos[0], e.pos[1], e.pos[2] + 1.75, e.vel, d)
        if tgt is None and self.goal is not None and self.goal[2] == "crate":
            for c in obs.get("crates") or ():
                if int(c["id"]) != self.goal[3]:
                    continue
                cx_, cy_, cz_ = c["pos"]
                d = math.hypot(cx_ - x, cy_ - y)
                if d < 110.0 and self.world.line_of_sight(muzzle, (cx_, cy_, cz_ + 0.8)):
                    tgt = (cx_, cy_, cz_ + 0.8, (0.0, 0.0, 0.0), d)
        if tgt is None:
            return None, (False, False)
        tx, ty, tz, vel, d = tgt
        tf = d / max(1.0, float(self.gun.get("speed", 252.0)))
        px, py = tx + vel[0] * tf, ty + vel[1] * tf
        yaw = math.atan2(py - y, px - x)
        pitch = clamp(math.atan2(tz - (z + self.muzzle_h), max(1.0, math.hypot(px - x, py - y))), self.min_pitch, self.max_pitch)
        fire0 = self._gun_ready() and self._aim_error(yaw, pitch) < clamp(math.atan2(1.4, max(8.0, d)), 0.006, 0.08)
        fire1 = False
        sec = self.weapons[1] if len(self.weapons) > 1 else None
        if sec is not None and sec.get("kind") == "lightning" and d < 40.0 and self.energy > 30.0 \
                and abs(angle_diff(self.self_aim[0], yaw)) < 0.2:
            fire1 = True
        return (yaw, pitch), (fire0, fire1)

    def _mortar_clear(self, aim: Vec2) -> bool:
        """The first 25 m of the shell's arc must not run into a wall."""
        x, y, z = self.self_pos
        yaw, pitch = aim
        v = float(self.mortar.get("speed", 72.0))
        g = float(self.mortar.get("gravity", 22.0))
        p0 = (x, y, z + self.height + 0.5)
        prev = p0
        for i in range(1, 6):
            t = i * 0.12
            h = v * math.cos(pitch) * t
            pt = (x + math.cos(yaw) * h, y + math.sin(yaw) * h, p0[2] + v * math.sin(pitch) * t - 0.5 * g * t * t)
            if self.world.segment_blocked(prev, pt):
                return False
            prev = pt
        return True

    def _gun_ready(self) -> bool:
        w = self.weapons[0] if self.weapons else None
        if not w:
            return False
        if float(w.get("reload", 0.0)) > 0.0:
            return False
        if float(w.get("ammo", 0.0)) <= 0.0:
            return False
        return float(w.get("cooldown", 0.0)) <= 0.01

    def _aim_point(self, target: Enemy, weapon: str, d: float) -> Optional[Vec2]:
        x, y, z = self.self_pos
        muzzle = z + self.muzzle_h
        tx, ty, tz = target.pos
        scale = target.scale
        if weapon == "gun":
            speed = float(self.gun.get("speed", 252.0))
            tf = d / max(1.0, speed)
            px = tx + target.vel[0] * tf
            py = ty + target.vel[1] * tf
            pz = tz + target.vel[2] * tf + 4.5 * scale
            return (math.atan2(py - y, px - x), clamp(math.atan2(pz - muzzle, max(1.0, math.hypot(px - x, py - y))), self.min_pitch, self.max_pitch))
        if weapon == "laser":
            # Hitscan: aim at the current centre.  A small lead compensates for torso slew.
            px = tx + target.vel[0] * 0.04
            py = ty + target.vel[1] * 0.04
            pz = tz + target.vel[2] * 0.04 + 4.5 * scale
            return (math.atan2(py - y, px - x), clamp(math.atan2(pz - muzzle, max(1.0, math.hypot(px - x, py - y))), self.min_pitch, self.max_pitch))
        if weapon == "lightning":
            px, py, pz = tx, ty, tz + 4.0 * scale
            return (math.atan2(py - y, px - x), clamp(math.atan2(pz - muzzle, max(1.0, math.hypot(px - x, py - y))), self.min_pitch, self.max_pitch))
        if weapon == "mortar":
            return self._mortar_aim(tx, ty, tz + 2.0 * scale, d)
        return None

    def _mortar_aim(self, tx: float, ty: float, tz: float, d: float, high: bool = False) -> Optional[Vec2]:
        x, y, z = self.self_pos
        muzzle = z + self.muzzle_h + 2.5
        v = float(self.mortar.get("speed", 72.0))
        g = float(self.mortar.get("gravity", 22.0))
        if d < 20.0 or d > 235.0:
            return None
        # iterate once for a moving target
        px, py, pz = tx, ty, tz
        pitch = None
        for _ in range(2):
            h = pz - muzzle
            vv = v * v
            disc = vv * vv - g * (g * d * d + 2.0 * h * vv)
            if disc < 0.0:
                return None
            root = math.sqrt(disc)
            u1 = (vv - root) / (g * d)
            u2 = (vv + root) / (g * d)
            u = u2 if high else (u1 if u1 > 0.05 else u2)
            pitch = math.atan(u)
            if pitch < self.min_pitch or pitch > self.max_pitch:
                return None
            tof = d / max(1.0, v * math.cos(pitch))
            # target velocity is not passed here; caller can pre-lead, so keep this simple
            break
        if pitch is None:
            return None
        return (math.atan2(py - y, px - x), clamp(pitch, self.min_pitch, self.max_pitch))

    def _aim_error(self, yaw: float, pitch: float) -> float:
        ey = abs(angle_diff(self.self_aim[0], yaw))
        ep = abs(self.self_aim[1] - pitch)
        return max(ey, ep * 1.15)

    def _scan_aim(self, obs: dict, move: Vec2) -> Vec2:
        x, y, z = self.self_pos
        # Face the direction of travel, with a slow scan to the sides when idle.
        if math.hypot(move[0], move[1]) > 0.15:
            yaw = math.atan2(move[1], move[0])
        else:
            yaw = self.self_aim[0]
        # If we were damaged from a bearing, look that way.
        if self.last_damage_bearing is not None and self.t - self.last_damage_t < 2.0:
            yaw = float(self.last_damage_bearing)
        # If a nearby sound is the only cue, look toward it.
        elif self.t - self.last_seen_enemy_t > 2.0 and self.sound_threats:
            recent = [s for s in self.sound_threats if self.t - s[0] < 1.5 and s[2] in ("gun", "laser", "lightning", "mortar", "explosion")]
            if recent:
                s = recent[-1]
                yaw = math.atan2(s[1][1] - y, s[1][0] - x)
        # Gentle scan while idle so we are not blind behind us.
        if math.hypot(move[0], move[1]) < 0.1:
            yaw += 0.45 * math.sin(self.t * 0.9 + self.slot)
        pitch = 0.0
        return (norm_angle(yaw), pitch)

    def _should_reload(self, obs: dict) -> bool:
        w = self.weapons[0] if self.weapons else None
        if not w:
            return False
        ammo = float(w.get("ammo", 0.0))
        reload = float(w.get("reload", 0.0))
        if reload > 0.0:
            return False
        if ammo <= 2.0:
            enemy, edist, n120, n60 = self._threat_summary()
            if edist > 140.0 or ammo <= 0.0:
                return True
        return False

    # ------------------------------------------------------------------ pickups / upgrades
    def _take_decision(self, obs: dict) -> Tuple[Optional[int], Optional[int]]:
        x, y, z = self.self_pos
        sec = self.weapons[1] if len(self.weapons) > 1 else None
        reach = float(self.loot.get("pickup_radius", 4.0))
        best = None
        best_score = -1.0
        for p in obs.get("pickups") or ():
            kind = p.get("kind")
            if kind not in ("laser", "lightning", "mortar"):
                continue
            px, py, pz = p["pos"]
            if math.hypot(px - x, py - y) > reach or abs(pz - z) > 4.0:
                continue
            if not self._want_weapon(kind):
                continue
            score = {"laser": 3.0, "lightning": 2.5, "mortar": 1.0}.get(kind, 0.0)
            if score > best_score:
                best_score = score
                best = p
        if best is None:
            return None, None
        drop = None
        if sec is not None:
            drop = 1
        return int(best["id"]), drop

    WEAPON_RANK = {"laser": 3, "lightning": 2, "mortar": 1}

    def _want_weapon(self, kind: str) -> bool:
        if kind not in self.WEAPON_RANK:
            return False
        sec = self.weapons[1] if len(self.weapons) > 1 else None
        if sec is None:
            return True
        if sec.get("kind") == "laser" and float(sec.get("charge", 0.0)) > 0.0:
            return False
        return self.WEAPON_RANK[kind] > self.WEAPON_RANK.get(sec.get("kind"), 0)

    def _choose_upgrade(self, offer: dict) -> int:
        oid = offer.get("id")
        self.last_offer_id = oid
        choices = offer.get("choices") or ()
        if not choices:
            return 0
        held = set(self._secondary_kinds())
        held.add("gun")
        scores = {
            "armor": 110.0, "plating": 100.0, "nanites": 90.0, "servos": 78.0,
            "gyros": 68.0, "sensors": 62.0, "reactor": 55.0, "capacitor": 48.0, "jets": 42.0,
            "gun_damage": 82.0, "gun_rate": 78.0, "gun_velocity": 62.0, "gun_mag": 56.0,
            "laser_damage": 96.0, "laser_charge": 86.0, "laser_pierce": 74.0,
            "lightning_efficiency": 96.0, "lightning_range": 84.0, "lightning_chain": 72.0,
            "mortar_splash": 80.0, "mortar_reload": 76.0, "mortar_velocity": 64.0,
        }
        best_i = 0
        best_s = -1.0
        for i, c in enumerate(choices):
            cid = c.get("id", "")
            s = scores.get(cid, 30.0)
            req = c.get("requires")
            if req and req not in held:
                s = -1.0
            if s > best_s:
                best_s = s
                best_i = i
        return best_i

    # ------------------------------------------------------------------ debug
    def _draw(self, obs: dict, aim: Vec2, mode: str) -> None:
        x, y, z = self.self_pos
        mf.draw.label(f"{mode} hp{self.hp:.0f} e{self.energy:.0f} c{self.cubes:.0f}")
        mf.draw.point([x, y, z + 2], "#40c0ff", 5)
        if self.goal is not None:
            mf.draw.line([x, y, z + 2], [self.goal[0], self.goal[1], z + 2], "#ffd060")
            mf.draw.point([self.goal[0], self.goal[1], z + 2], "#ffd060", 6)
        if self.path:
            for i in range(self.path_i, min(len(self.path), self.path_i + 6)):
                px, py = self.path[i]
                mf.draw.point([px, py, self.world.height(px, py) + 1], "#80ff80", 3)
        enemy, edist, n120, n60 = self._threat_summary()
        if enemy is not None:
            mf.draw.line([x, y, z + 7], [enemy.pos[0], enemy.pos[1], enemy.pos[2] + 4.5], "#ff5040")


if __name__ == "__main__":
    mf.run(Bot)
