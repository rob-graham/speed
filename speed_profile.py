"""Utilities for generating a motorcycle speed profile along a racing line.

The script reads a CSV file containing ``x,y`` coordinates that describe the
centre line of a race track.  The path is resampled to roughly equal spacing
and a simple physics model is applied to estimate the maximum speed the bike
can maintain at each point.  The resulting profile including gear selection and
engine RPM is written to a new CSV file.

Run ``python speed_profile.py --help`` for a list of available command line
options.
"""

import argparse
import csv
from dataclasses import dataclass
from typing import List, Tuple
import math


def load_csv(path: str) -> List[Tuple[float, float]]:
    """Load x,y coordinates from a CSV file with two columns and no header."""
    pts: List[Tuple[float, float]] = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            x, y = map(float, row[:2])
            pts.append((x, y))
    return pts


def cumulative_distance(pts: List[Tuple[float, float]]) -> List[float]:
    """Return cumulative distance along *pts* starting from zero."""
    s = [0.0]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        s.append(s[-1] + math.hypot(x1 - x0, y1 - y0))
    return s

def save_csv(path: str,
             pts: List[Tuple[float, float]],
             dists: List[float],
             speeds: List[float],
             gears: List[int],
             rpms: List[float]) -> None:
    """Save x,y coordinates with distance, speed (m/s and kph), gear and rpm to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m", "y_m", "s_m", "v_mps", "v_kph", "gear", "rpm"])
        for (x, y), s, v, g, r in zip(pts, dists, speeds, gears, rpms):
            writer.writerow([x, y, s, v, v * 3.6, g, r])

def _arc_segment(p0: Tuple[float, float],
                 p1: Tuple[float, float],
                 p2: Tuple[float, float],
                 step: float) -> List[Tuple[float, float]]:
    """Return points from *p1* to *p2* following a circular arc through
    (*p0*, *p1*, *p2*).  If the points are nearly collinear a straight line is
    generated instead."""
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    area = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    if abs(area) < 1e-9:
        seg_len = math.hypot(x3 - x2, y3 - y2)
        n = max(1, int(math.floor(seg_len / step)))
        return [
            (x2 + (j / n) * (x3 - x2), y2 + (j / n) * (y3 - y2))
            for j in range(n + 1)
        ]

    b = (x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (
        x3 * x3 + y3 * y3
    ) * (y2 - y1)
    c = (x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (
        x3 * x3 + y3 * y3
    ) * (x1 - x2)
    cx = -b / (2 * area)
    cy = -c / (2 * area)
    r = math.hypot(x2 - cx, y2 - cy)
    th1 = math.atan2(y2 - cy, x2 - cx)
    th2 = math.atan2(y3 - cy, x3 - cx)
    cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
    if cross > 0.0:
        if th2 <= th1:
            th2 += 2 * math.pi
    else:
        if th2 >= th1:
            th2 -= 2 * math.pi
    arc_len = r * abs(th2 - th1)
    n = max(1, int(math.floor(arc_len / step)))
    return [
        (cx + r * math.cos(th1 + (j / n) * (th2 - th1)),
         cy + r * math.sin(th1 + (j / n) * (th2 - th1)))
        for j in range(n + 1)
    ]


def resample(points: List[Tuple[float, float]], step: float) -> List[Tuple[float, float]]:
    """Resample *points* using circular arcs so that adjacent points are
    approximately *step* metres apart."""
    if len(points) < 3:
        return points

    if math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) > 1e-6:
        points = points + [points[0]]

    resampled: List[Tuple[float, float]] = []
    n = len(points) - 1  # last equals first
    for i in range(n):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[(i + 1) % n]
        seg = _arc_segment(p_prev, p_curr, p_next, step)
        if i == 0:
            resampled.extend(seg)
        else:
            resampled.extend(seg[1:])

    resampled.append(resampled[0])
    return resampled

def _curvature(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Return curvature (1/radius) for three consecutive points."""
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    a = math.hypot(x2 - x1, y2 - y1)
    b = math.hypot(x3 - x2, y3 - y2)
    c = math.hypot(x3 - x1, y3 - y1)
    area = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / 2.0
    if area == 0:
        return 0.0
    radius = a * b * c / (4.0 * area)
    return 1.0 / radius


@dataclass
class BikeParams:
    """Parameters describing the motorcycle and environment.

    The defaults approximate a 600-class sport bike.  Adjust these values to
    model different machines or track conditions.  All forces are expressed in
    SI units.
    """

    rho: float = 1.225
    g: float = 9.81
    m: float = 265.0
    CdA: float = 0.35
    Crr: float = 0.015
    rw: float = 0.31
    mu: float = 1.20
    a_wheelie_max: float = 1.0 * 9.81   # was 0.9 * 9.81
    a_brake: float = 1.2 * 9.81
    shift_rpm: float = 16000.0
    primary: float = 85.0/41.0
    final_drive: float = 47.0/16.0
    gears: Tuple[float, ...] = (2.583, 2.000, 1.667, 1.444, 1.286, 1.150)
    eta_driveline: float = 0.95
    T_peak: float = 63.0

    def torque_curve(self, rpm: float) -> float:
        """Simple flat torque curve."""
        return self.T_peak


def engine_rpm(v: float, bp: BikeParams, gear: float) -> float:
    omega_w = v / bp.rw
    omega_e = omega_w * bp.primary * bp.final_drive * gear
    return omega_e * 60.0 / (2 * math.pi)


def wheel_force(v: float, bp: BikeParams, gear: float) -> float:
    T = bp.torque_curve(engine_rpm(v, bp, gear))
    G = bp.primary * bp.final_drive * gear
    return (T * G * bp.eta_driveline) / bp.rw


def aero_drag(v: float, bp: BikeParams) -> float:
    return 0.5 * bp.rho * bp.CdA * v * v


def roll_res(bp: BikeParams) -> float:
    return bp.Crr * bp.m * bp.g


def traction_circle_cap(v: float, radius: float, bp: BikeParams, eps: float = 0.0) -> float:
    radius = max(radius, 1.0)
    a_lat = (v * v) / radius
    inside = (bp.mu * bp.g) ** 2 * (1.0 - eps) - a_lat * a_lat
    return math.sqrt(max(0.0, inside))


def select_gear(v: float, bp: BikeParams) -> float:
    """Select the highest gear that keeps engine rpm below the shift point."""
    for g in bp.gears:
        if engine_rpm(v, bp, g) <= bp.shift_rpm:
            return g
    return bp.gears[-1]

def compute_speed_profile(
    pts: List[Tuple[float, float]],
    bp: BikeParams,
    use_traction_circle: bool = False,
    trail_braking: bool = False,
    sweeps: int = 8,
    curv_smoothing: int = 3,
    speed_smoothing: int = 3,
) -> Tuple[List[float], float]:
    """Compute a speed profile and lap time for *pts*.

    The solver performs a number of forward and backward sweeps along the
    resampled path, applying acceleration limits from the engine, aerodynamics
    and optional traction circle.  When ``trail_braking`` is true the same
    traction limit is applied while decelerating.  ``curv_smoothing`` and
    ``speed_smoothing`` control how many neighbour-averaging passes are applied
    to the corner radius and final speeds respectively, which can help remove
    jitter on high resolution tracks.
    The function returns a list of speeds in metres per second for each path
    point and the overall lap time in seconds.
    """

    n = len(pts)
    if n < 3:
        return [0.0] * n, 0.0

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    ds = [math.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(n - 1)]
    R = [0.0] * n
    for i in range(1, n - 1):
        curv = _curvature(pts[i - 1], pts[i], pts[i + 1])
        R[i] = 1.0 / max(abs(curv), 1e-9)
    R[0] = R[1]
    R[-1] = R[-2]
    for _ in range(curv_smoothing):
        R_s = R.copy()
        for i in range(n):
            R_s[i] = 0.25 * R[(i - 1) % n] + 0.5 * R[i] + 0.25 * R[(i + 1) % n]
        R = R_s
        
    v = [math.sqrt(max(0.0, bp.mu * bp.g * max(R[i], 1.0))) * 0.97 for i in range(n)]

    for _ in range(sweeps):
        for i in range(n - 1):
            v_i = v[i]
            gear = select_gear(v_i, bp)
            Fw = wheel_force(v_i, bp, gear)
            a_pow = (Fw - aero_drag(v_i, bp) - roll_res(bp)) / bp.m
            a_pow = min(a_pow, bp.a_wheelie_max)
            a_pow = max(a_pow, -bp.a_brake)
            if use_traction_circle:
                a_pow = min(a_pow, traction_circle_cap(v_i, R[i], bp))
            v_next = math.sqrt(max(0.0, v_i * v_i + 2 * a_pow * ds[i]))
            if v_next < v[i + 1]:
                v[i + 1] = v_next
        for i in range(n - 2, -1, -1):
            a_brk = bp.a_brake
            if use_traction_circle and trail_braking:
                a_brk = min(a_brk, traction_circle_cap(v[i + 1], R[i + 1], bp))
            v_prev = math.sqrt(max(0.0, v[i + 1] * v[i + 1] + 2 * a_brk * ds[i]))
            if v_prev < v[i]:
                v[i] = v_prev
        v_loop = min(v[0], v[-1])
        v[0] = v[-1] = v_loop

    # simple neighbour averaging to damp residual jitter
    for _ in range(speed_smoothing):
        v_smooth = v.copy()
        for i in range(n):
            v_smooth[i] = 0.25 * v[(i - 1) % n] + 0.5 * v[i] + 0.25 * v[(i + 1) % n]
        v = v_smooth
        v_loop = min(v[0], v[-1])
        v[0] = v[-1] = v_loop

    lap_time = 0.0
    for i in range(n - 1):
        v_avg = max(1e-3, 0.5 * (v[i] + v[i + 1]))
        lap_time += ds[i] / v_avg

    return v, lap_time


@dataclass
class Segment:
    """Base class for track segments."""
    def points(self, x: float, y: float, heading: float, step: float) -> Tuple[List[Tuple[float, float]], float, float, float]:
        raise NotImplementedError

@dataclass
class Straight(Segment):
    length: float
    def points(self, x: float, y: float, heading: float, step: float):
        pts: List[Tuple[float, float]] = []
        n = max(1, int(math.ceil(self.length / step)))
        ds = self.length / n
        for _ in range(n):
            x += ds * math.cos(heading)
            y += ds * math.sin(heading)
            pts.append((x, y))
        return pts, x, y, heading

@dataclass
class Arc(Segment):
    radius: float
    angle: float  # radians, positive for left turn, negative for right
    def points(self, x: float, y: float, heading: float, step: float):
        pts: List[Tuple[float, float]] = []
        arc_len = abs(self.radius * self.angle)
        n = max(1, int(math.ceil(arc_len / step)))
        ds = arc_len / n
        sign = 1.0 if self.angle >= 0 else -1.0
        for _ in range(n):
            heading += sign * ds / self.radius
            x += ds * math.cos(heading)
            y += ds * math.sin(heading)
            pts.append((x, y))
        return pts, x, y, heading

def build_track(segments: List[Segment], step: float = 1.0,
                 start: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[Tuple[float, float]]:
    """Generate track points from straight and arc segments."""
    x, y, heading = start
    pts = [(x, y)]
    for seg in segments:
        new_pts, x, y, heading = seg.points(x, y, heading, step)
        pts.extend(new_pts)
    return pts

def _parse_gears(s: str) -> Tuple[float, ...]:
    return tuple(float(g) for g in s.split(",") if g)


def main():
    parser = argparse.ArgumentParser(description="Generate speed profile for a track")
    parser.add_argument("input", help="Input CSV with x,y points")
    parser.add_argument("output", help="Output CSV with x,y,v")
    parser.add_argument("--step", type=float, default=2.0, help="Resampling distance in metres")

    # Motorcycle and environment parameters
    parser.add_argument("--rho", type=float, default=1.225)
    parser.add_argument("--g", type=float, default=9.81)
    parser.add_argument("--m", type=float, default=265.0)
    parser.add_argument("--CdA", type=float, default=0.35)
    parser.add_argument("--Crr", type=float, default=0.015)
    parser.add_argument("--rw", type=float, default=0.31)
    parser.add_argument("--mu", type=float, default=1.20)
    parser.add_argument("--a_wheelie_max", type=float, default=1.0 * 9.81)
    parser.add_argument("--a_brake", type=float, default=1.2 * 9.81)
    parser.add_argument("--shift_rpm", type=float, default=16000.0)
    parser.add_argument("--primary", type=float, default=85.0/41.0)
    parser.add_argument("--final_drive", type=float, default=47.0/16.0)
    parser.add_argument("--gears", type=str, default="2.583, 2.000, 1.667, 1.444, 1.286, 1.150",
                        help="Comma separated gear ratios")
    parser.add_argument("--eta_driveline", type=float, default=0.95)
    parser.add_argument("--T_peak", type=float, default=63.0)

    parser.add_argument("--traction-circle", action="store_true",
                        help="Enable traction circle for acceleration")
    parser.add_argument("--trail-braking", action="store_true",
                        help="Apply traction circle during braking")
    parser.add_argument("--sweeps", type=int, default=25,
                        help="Forward/back sweeps for solver")
    parser.add_argument("--curv-smooth", type=int, default=3,
                        help="Neighbour-averaging passes for corner radius")
    parser.add_argument("--speed-smooth", type=int, default=3,
                        help="Neighbour-averaging passes for final speeds")
    
    args = parser.parse_args()

    pts = load_csv(args.input)
    pts = resample(pts, args.step)
    dists = cumulative_distance(pts)
    
    bp = BikeParams(
        rho=args.rho,
        g=args.g,
        m=args.m,
        CdA=args.CdA,
        Crr=args.Crr,
        rw=args.rw,
        mu=args.mu,
        a_wheelie_max=args.a_wheelie_max,
        a_brake=args.a_brake,
        shift_rpm=args.shift_rpm,
        primary=args.primary,
        final_drive=args.final_drive,
        gears=_parse_gears(args.gears),
        eta_driveline=args.eta_driveline,
        T_peak=args.T_peak,
    )

    speeds, lap_time = compute_speed_profile(
        pts,
        bp,
        use_traction_circle=args.traction_circle,
        trail_braking=args.trail_braking,
        sweeps=args.sweeps,
        curv_smoothing=args.curv_smooth,
        speed_smoothing=args.speed_smooth,
    )
    gears: List[int] = []
    rpms: List[float] = []
    for v in speeds:
        gear_ratio = select_gear(v, bp)
        gears.append(bp.gears.index(gear_ratio) + 1)
        rpms.append(engine_rpm(v, bp, gear_ratio))
        
    save_csv(args.output, pts, dists, speeds, gears, rpms)
    print(f"Lap time: {lap_time:.2f} s")

if __name__ == "__main__":
    main()