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

def save_csv(path: str, pts: List[Tuple[float, float]], speeds: List[float]) -> None:
    """Save x,y coordinates with speed to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m", "y_m", "v_mps"])
        for (x, y), v in zip(pts, speeds):
            writer.writerow([x, y, v])

def resample(points: List[Tuple[float, float]], step: float) -> List[Tuple[float, float]]:
    """Resample a list of points so that adjacent points are spaced by *step* metres.
    Linear interpolation is used between input points."""
    if len(points) < 2:
        return points
    pts: List[Tuple[float, float]] = []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.floor(seg_len / step)))
        for j in range(n):
            t = j / n
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    pts.append(points[-1])
    return pts

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

def compute_speed_profile(pts: List[Tuple[float, float]], a_lat: float = 12.0,
                          a_acc: float = 3.0, a_brake: float = 6.0,
                          v_max_straight: float = 100.0) -> List[float]:
    """Compute a smooth speed profile for *pts*.

    Parameters
    ----------
    pts : list of (x, y)
        Track centreline points in metres.
    a_lat : float
        Maximum lateral acceleration (m/s^2) allowed.
    a_acc : float
        Maximum longitudinal acceleration (m/s^2).
    a_brake : float
        Maximum braking deceleration (m/s^2).
    v_max_straight : float
        Optional cap on straight line speed (m/s).
    """
    n = len(pts)
    if n < 3:
        return [0.0] * n
    ds = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
          for i in range(n - 1)]
    curv = [0.0] * n
    for i in range(1, n - 1):
        curv[i] = _curvature(pts[i - 1], pts[i], pts[i + 1])
    v_max = [v_max_straight] * n
    for i in range(n):
        if curv[i] > 1e-9:
            v_max[i] = math.sqrt(a_lat / curv[i])
    v = v_max[:]
    for i in range(1, n):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_acc * ds[i - 1]))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_brake * ds[i]))
    return v

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

def main():
    parser = argparse.ArgumentParser(description="Generate speed profile for a track")
    parser.add_argument("input", help="Input CSV with x,y points")
    parser.add_argument("output", help="Output CSV with x,y,v")
    parser.add_argument("--step", type=float, default=2.0, help="Resampling distance in metres")
    args = parser.parse_args()
    pts = load_csv(args.input)
    pts = resample(pts, args.step)
    speeds = compute_speed_profile(pts)
    save_csv(args.output, pts, speeds)

if __name__ == "__main__":
    main()