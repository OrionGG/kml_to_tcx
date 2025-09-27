#!/usr/bin/env python3
"""
estimate_twd_improved.py

Improved single-file estimator of TRUE WIND DIRECTION (TWD) from a TCX GPS track.
This version is pure-Python (no pandas/numpy) and automatically attempts
multiple filtering strategies when the track is mixed (beating + reaching + downwind).

Update: added automatic detection of steady tack segments and a "segment-pair"
strategy that pairs steady starboard/port runs to compute a bisector — this
often greatly improves confidence when a clear upwind leg exists but is
buried among reaching or downwind legs.

Usage:
    python estimate_twd_improved.py path/to/track.tcx [--plot]

"""

import sys
import math
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import statistics

# optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# ---------------- geometry helpers ----------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))


def bearing_deg(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    theta = math.degrees(math.atan2(x, y))
    return (theta + 360.0) % 360.0


def angular_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def angular_distance(a, b):
    return abs(angular_diff(a, b))


def circular_mean_deg(angles):
    if not angles:
        return 0.0
    sx = 0.0; cx = 0.0
    for a in angles:
        r = math.radians(a % 360.0)
        sx += math.sin(r); cx += math.cos(r)
    sx /= len(angles); cx /= len(angles)
    if abs(sx) < 1e-12 and abs(cx) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(sx, cx)) % 360.0


def circular_std_deg(angles):
    if not angles:
        return 0.0
    sx = 0.0; cx = 0.0
    for a in angles:
        r = math.radians(a % 360.0)
        sx += math.sin(r); cx += math.cos(r)
    sx /= len(angles); cx /= len(angles)
    R = math.hypot(sx, cx)
    R = max(min(R, 1.0), 1e-12)
    std_rad = math.sqrt(max(0.0, -2.0 * math.log(R)))
    return math.degrees(std_rad)

# ---------------- TCX parsing ----------------

def parse_tcx_positions(tcx_path):
    ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
    tree = ET.parse(tcx_path)
    root = tree.getroot()
    trkpts = []
    for tp in root.findall('.//tcx:Trackpoint', ns):
        time_el = tp.find('tcx:Time', ns)
        pos_el = tp.find('tcx:Position', ns)
        if time_el is None or pos_el is None:
            continue
        lat_el = pos_el.find('tcx:LatitudeDegrees', ns)
        lon_el = pos_el.find('tcx:LongitudeDegrees', ns)
        if lat_el is None or lon_el is None:
            continue
        t = time_el.text
        try:
            dt = datetime.fromisoformat(t.replace('Z', '+00:00'))
        except Exception:
            try:
                dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            except Exception:
                continue
        lat = float(lat_el.text); lon = float(lon_el.text)
        trkpts.append({'time': dt, 'lat': lat, 'lon': lon})
    if not trkpts:
        raise ValueError("No trackpoints with lat/lon found in the TCX.")
    trkpts.sort(key=lambda x: x['time'])
    return trkpts

# ---------------- compute motion ----------------

def compute_motion_list(points):
    out = []
    prev = None
    for p in points:
        rec = dict(p)
        rec['dt_s'] = 0.0
        rec['dist_m'] = 0.0
        rec['speed_m_s'] = 0.0
        rec['cog_deg'] = float('nan')
        if prev is not None:
            dt = (p['time'] - prev['time']).total_seconds()
            if dt < 0:
                dt = 0.0
            rec['dt_s'] = dt
            d = haversine_m(prev['lat'], prev['lon'], p['lat'], p['lon'])
            rec['dist_m'] = d
            rec['speed_m_s'] = d / dt if dt > 0 else 0.0
            rec['cog_deg'] = bearing_deg(prev['lat'], prev['lon'], p['lat'], p['lon'])
        out.append(rec)
        prev = p
    return out

# ---------------- circular k-means (k=2) ----------------

def circular_kmeans_two(angles, max_iter=200, tol=1e-3):
    if len(angles) < 2:
        return None, None
    angles = [a % 360.0 for a in angles]
    sorted_a = sorted(angles)
    a0 = sorted_a[max(0, int(0.25*len(sorted_a)))]
    a1 = sorted_a[min(len(sorted_a)-1, int(0.75*len(sorted_a)))]
    centers = [a0, a1]
    for _ in range(max_iter):
        clusters = {0: [], 1: []}
        for a in angles:
            d0 = angular_distance(a, centers[0])
            d1 = angular_distance(a, centers[1])
            k = 0 if d0 <= d1 else 1
            clusters[k].append(a)
        new_centers = []
        changed = False
        for k in (0,1):
            if clusters[k]:
                cm = circular_mean_deg(clusters[k])
            else:
                cm = centers[k]
            new_centers.append(cm)
            if angular_distance(cm, centers[k]) > tol:
                changed = True
        centers = new_centers
        if not changed:
            break
    labels = []
    for a in angles:
        d0 = angular_distance(a, centers[0])
        d1 = angular_distance(a, centers[1])
        labels.append(0 if d0 <= d1 else 1)
    return centers, labels

# ---------------- smoothing ----------------

def circular_moving_average(angles, window):
    if window <= 1:
        return angles[:]
    half = window // 2
    n = len(angles)
    sin_vals = [math.sin(math.radians(a)) for a in angles]
    cos_vals = [math.cos(math.radians(a)) for a in angles]
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        s = statistics.mean(sin_vals[lo:hi])
        c = statistics.mean(cos_vals[lo:hi])
        ang = math.degrees(math.atan2(s, c)) % 360.0
        smoothed.append(ang)
    return smoothed

# ---------------- probabilistic score for bisector candidates ----------------

def bisector_prob_score(headings_deg, wind_candidate_deg):
    if not headings_deg:
        return 0.0
    r = [angular_diff(h, wind_candidate_deg) for h in headings_deg]
    pos = [abs(x) for x in r if x > 0]
    neg = [abs(x) for x in r if x < 0]
    if not pos or not neg:
        overall_std = circular_std_deg([ (x + 360.0) % 360.0 for x in r ])
        return max(0.0, 0.3 * math.exp(-overall_std/60.0))
    mu_pos = statistics.mean(pos)
    mu_neg = statistics.mean(neg)
    symmetry = 1.0 / (1.0 + abs(mu_pos - mu_neg))
    overall_std = circular_std_deg([ (x + 360.0) % 360.0 for x in r ])
    concentration = math.exp(-overall_std / 60.0)
    sep = mu_pos + mu_neg
    ideal_sep = 80.0
    sep_score = math.exp(-((sep - ideal_sep)**2) / (2 * (30.0**2)))
    score = 0.45 * symmetry + 0.35 * concentration + 0.20 * sep_score
    return float(max(0.0, min(1.0, score)))

# ---------------- upwind-likelihood detector ----------------

def upwind_likelihood_score(headings_deg, wind_candidate_deg, target_close_hauled=45.0):
    if not headings_deg:
        return 0.0
    r = [angular_diff(h, wind_candidate_deg) for h in headings_deg]
    pos = [abs(x) for x in r if x > 0]
    neg = [abs(x) for x in r if x < 0]
    if not pos or not neg:
        return 0.0
    mu_pos = statistics.mean(pos);
    mu_neg = statistics.mean(neg)
    sigma = 15.0
    closeness = math.exp(-((mu_pos - target_close_hauled)**2 + (mu_neg - target_close_hauled)**2) / (2 * sigma * sigma))
    symmetry = 1.0 / (1.0 + abs(mu_pos - mu_neg))
    npos = len(pos); nneg = len(neg)
    balance = min(npos, nneg) / max(npos, nneg)
    sep = mu_pos + mu_neg
    if sep > 140:
        sep_penalty = 0.5
    else:
        sep_penalty = 1.0
    score = closeness * symmetry * balance * sep_penalty
    return float(max(0.0, min(1.0, score)))

# ---------------- steady-segment detection ----------------

def detect_steady_segments(motion, min_points=8, max_turn_deg=12.0):
    """
    Detect contiguous runs where heading changes between consecutive samples are small
    (|delta heading| <= max_turn_deg). Returns list of segments; each segment is a list
    of motion records. Intended to find steady-starboard/starboard runs (tack runs).
    """
    segments = []
    if not motion:
        return segments
    cur = [motion[0]]
    for prev, currec in zip(motion, motion[1:]):
        if math.isnan(prev.get('cog_deg', float('nan'))) or math.isnan(currec.get('cog_deg', float('nan'))):
            # break segment
            if len(cur) >= min_points:
                segments.append(cur)
            cur = [currec]
            continue
        d = abs(angular_diff(currec['cog_deg'], prev['cog_deg']))
        if d <= max_turn_deg:
            cur.append(currec)
        else:
            if len(cur) >= min_points:
                segments.append(cur)
            cur = [currec]
    if len(cur) >= min_points:
        segments.append(cur)
    return segments

# ---------------- estimator (core) ----------------

def run_estimator_on_motion(motion, min_points=10, smoothing_frac=0.02):
    angles = [m['cog_deg'] for m in motion if not math.isnan(m['cog_deg'])]
    if len(angles) < min_points:
        return None
    window = max(3, int(len(angles) * smoothing_frac))
    if window % 2 == 0:
        window += 1
    smoothed = circular_moving_average(angles, window)
    centers, labels = circular_kmeans_two(smoothed)
    if centers is None:
        return None
    c1, c2 = centers[0], centers[1]
    sep = angular_distance(c1, c2)
    bis = circular_mean_deg([c1, c2])
    bis2 = (bis + 180.0) % 360.0
    s1 = bisector_prob_score(smoothed, bis)
    s2 = bisector_prob_score(smoothed, bis2)
    if s2 > s1:
        wind_from = bis2
        chosen_score = s2
    else:
        wind_from = bis
        chosen_score = s1
    upwind_score_chosen = upwind_likelihood_score(smoothed, wind_from)
    cl0 = [smoothed[i] for i,lab in enumerate(labels) if lab == 0]
    cl1 = [smoothed[i] for i,lab in enumerate(labels) if lab == 1]
    spread1 = circular_std_deg(cl0) if cl0 else float('nan')
    spread2 = circular_std_deg(cl1) if cl1 else float('nan')
    overall_spread = circular_std_deg(smoothed)
    confidence = max(0.0, min(1.0, (sep - 10.0) / 100.0))
    confidence *= (1.0 - min(1.0, overall_spread / 60.0))
    weight_upwind = 0.9
    boosted_confidence = max(confidence, upwind_score_chosen * weight_upwind)
    final_score = 0.45 * chosen_score + 0.35 * upwind_score_chosen + 0.20 * boosted_confidence
    return {
        'wind_from_deg': float(wind_from % 360.0),
        'centers_deg': (float(c1), float(c2)),
        'separation_deg': float(sep),
        'spread_cluster1_deg': float(spread1) if not math.isnan(spread1) else None,
        'spread_cluster2_deg': float(spread2) if not math.isnan(spread2) else None,
        'overall_spread_deg': float(overall_spread),
        'confidence_0_1': float(boosted_confidence),
        'bisector_prob': float(chosen_score),
        'upwind_likelihood': float(upwind_score_chosen),
        'final_score': float(final_score),
        'n_points_used': len(angles),
        'angles_smoothed': smoothed,
        'labels': labels
    }

# ---------------- high-level orchestration ----------------

def estimate_true_wind(points, args):
    motion_all = compute_motion_list(points)
    def apply_speed_filter(motion, min_speed, max_speed):
        return [m for m in motion if m['speed_m_s'] >= min_speed and m['speed_m_s'] <= max_speed and not math.isnan(m['cog_deg'])]

    candidates = []
    baseline = apply_speed_filter(motion_all, args.min_speed, args.max_speed)
    res_full = run_estimator_on_motion(baseline, min_points=args.min_points)
    if res_full:
        candidates.append(('full', res_full, baseline))

    # detect steady segments (tack runs) and pair complementary segments
    segments = detect_steady_segments(baseline, min_points=max(6, args.min_points), max_turn_deg=12.0)
    # compute mean heading for each segment
    seg_means = [(i, circular_mean_deg([m['cog_deg'] for m in seg]), seg) for i, seg in enumerate(segments)]
    # pair segments with opposite sides and reasonable separation (40..140 deg)
    for i, mean_i, seg_i in seg_means:
        for j, mean_j, seg_j in seg_means:
            if j <= i:
                continue
            sep = angular_distance(mean_i, mean_j)
            if 40 <= sep <= 140:
                combined = seg_i + seg_j
                res_pair = run_estimator_on_motion(combined, min_points=args.min_points)
                if res_pair:
                    candidates.append((f'segment_pair_{i}_{j}', res_pair, combined))

    spread_thresh = args.spread_threshold
    if res_full is None or (res_full and res_full['overall_spread_deg'] > spread_thresh):
        if args.mark_lat is not None and args.mark_lon is not None:
            subset = []
            for m in baseline:
                b_to_mark = bearing_deg(m['lat'], m['lon'], args.mark_lat, args.mark_lon)
                if angular_distance(m['cog_deg'], b_to_mark) <= args.max_off_angle:
                    subset.append(m)
            if len(subset) >= args.min_points:
                res_mark = run_estimator_on_motion(subset, min_points=args.min_points)
                if res_mark:
                    candidates.append(('mark_latlon', res_mark, subset))
        elif args.mark_bearing is not None:
            subset = [m for m in baseline if angular_distance(m['cog_deg'], args.mark_bearing) <= args.max_off_angle]
            if len(subset) >= args.min_points:
                res_mark = run_estimator_on_motion(subset, min_points=args.min_points)
                if res_mark:
                    candidates.append(('mark_bearing', res_mark, subset))

    if res_full is None or (res_full and res_full['overall_spread_deg'] > spread_thresh):
        beat_max = args.beat_max_speed
        subset_speed = apply_speed_filter(motion_all, args.min_speed, beat_max)
        if len(subset_speed) >= args.min_points:
            res_speed = run_estimator_on_motion(subset_speed, min_points=args.min_points)
            if res_speed:
                candidates.append(('beat_speed', res_speed, subset_speed))

    if not candidates:
        return None, motion_all

    def score_candidate(item):
        tag, res, subset = item
        sc = res.get('final_score', 0.0)
        n = res.get('n_points_used', 0)
        return sc + (min(n, 300) / 300.0) * 0.05

    best = max(candidates, key=score_candidate)
    return best, motion_all

# ---------------- plotting ----------------

def plot_result(subset_motion, res, title_extra=''):
    if not HAS_PLT:
        print('matplotlib not available; cannot plot.')
        return
    lats = [m['lat'] for m in subset_motion]
    lons = [m['lon'] for m in subset_motion]
    labels = res['labels']
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(lons, lats, c=labels, s=8, cmap='tab10')
    ax1.set_title('Track (subset) ' + title_extra)
    ax1.set_xlabel('Longitude'); ax1.set_ylabel('Latitude')
    ax1.axis('equal')
    ax2 = fig.add_subplot(1,2,2, projection='polar')
    angs = [math.radians(a) for a in res['angles_smoothed']]
    ax2.set_theta_zero_location('N'); ax2.set_theta_direction(-1)
    ax2.scatter(angs, [1.0]*len(angs), c=res['labels'], s=8)
    for c in res['centers_deg']:
        ax2.scatter(math.radians(c), 1.2, s=100, marker='x')
    wf = res['wind_from_deg']
    ax2.arrow(math.radians(wf), 0.0, 0, 1.4, width=0.03, alpha=0.6)
    ax2.set_title('Smoothed headings & wind-from\n' + title_extra)
    plt.tight_layout()
    plt.show()

# ---------------- utility ----------------

def deg_to_cardinal(d):
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    ix = int((d + 11.25) / 22.5) % 16
    return dirs[ix]

# ---------------- CLI ----------------

def main():
    p = argparse.ArgumentParser(description='Estimate TWD from TCX (improved, segmentation).')
    p.add_argument('tcx', help='Path to TCX file')
    p.add_argument('--min-speed', type=float, default=0.8, help='Min speed (m/s) to consider moving (default 0.8)')
    p.add_argument('--max-speed', type=float, default=100.0, help='Max speed (m/s) to accept (default large)')
    p.add_argument('--beat-max-speed', type=float, default=3.0, help='Max speed to consider as beating (m/s) when trying speed filter (default 3.0)')
    p.add_argument('--min-points', type=int, default=10, help='Minimum points needed for estimation')
    p.add_argument('--mark-lat', type=float, default=None, help='Latitude of a known mark (optional)')
    p.add_argument('--mark-lon', type=float, default=None, help='Longitude of a known mark (optional)')
    p.add_argument('--mark-bearing', type=float, default=None, help='Absolute bearing (deg) to known mark/windward (optional)')
    p.add_argument('--max-off-angle', type=float, default=60.0, help='Max angular offset (deg) from bearing-to-mark to select upwind points (default 60)')
    p.add_argument('--spread-threshold', type=float, default=60.0, help='If overall spread (deg) exceeds this, try alternative filters (default 60)')
    p.add_argument('--plot', action='store_true', help='Show diagnostic plots (requires matplotlib)')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    points = parse_tcx_positions(args.tcx)
    if not points:
        print('No positions found in TCX.', file=sys.stderr); sys.exit(2)

    best, motion_all = estimate_true_wind(points, args)
    if not best:
        print('Failed to estimate wind direction (not enough valid moving points).', file=sys.stderr)
        sys.exit(2)
    tag, res, subset = best

    print(f"Chosen strategy: {tag}")
    wf = res['wind_from_deg']
    print(f"Estimated TRUE WIND (wind FROM) direction: {wf:.1f}\u00b0 (0\u00b0=North, 90\u00b0=East)")
    print(f" - Cardinal approx.          : {deg_to_cardinal(wf)}")
    print(f" - Cluster centers (headings) : {res['centers_deg'][0]:.1f}\u00b0, {res['centers_deg'][1]:.1f}\u00b0")
    print(f" - Separation between centers : {res['separation_deg']:.1f}\u00b0")
    print(f" - Overall heading spread     : {res['overall_spread_deg']:.1f}\u00b0")
    print(f" - Cluster spreads (1/2)     : {res['spread_cluster1_deg']:.1f}\u00b0 / {res['spread_cluster2_deg']:.1f}\u00b0")
    print(f" - Points used               : {res['n_points_used']}")
    print(f" - Bisector probability score: {res['bisector_prob']:.3f}")
    print(f" - Upwind-likelihood score   : {res['upwind_likelihood']:.3f}")
    print(f" - Confidence (0..1)         : {res['confidence_0_1']:.3f}")
    print(f" - Final blended score       : {res['final_score']:.3f}")

    if args.verbose:
        print('\nAll candidate strategies tried:')
        for tag_c, r_c, s_c in [ (t,r,s) for (t,r,s) in [( 'full', run_estimator_on_motion([m for m in motion_all if m['speed_m_s'] >= args.min_speed and m['speed_m_s'] <= args.max_speed], min_points=args.min_points), None)] if r is not None]:
            pass
        # recompute and show candidates more simply
        baseline_motion = [m for m in motion_all if m['speed_m_s'] >= args.min_speed and m['speed_m_s'] <= args.max_speed]
        candidates_debug = []
        r = run_estimator_on_motion(baseline_motion, min_points=args.min_points)
        if r:
            candidates_debug.append(('full', r))
        segments = detect_steady_segments(baseline_motion, min_points=max(6, args.min_points), max_turn_deg=12.0)
        seg_means = [(i, circular_mean_deg([m['cog_deg'] for m in seg]), seg) for i, seg in enumerate(segments)]
        for i, mean_i, seg_i in seg_means:
            for j, mean_j, seg_j in seg_means:
                if j <= i:
                    continue
                sep = angular_distance(mean_i, mean_j)
                if 40 <= sep <= 140:
                    combined = seg_i + seg_j
                    r_pair = run_estimator_on_motion(combined, min_points=args.min_points)
                    if r_pair:
                        candidates_debug.append((f'segment_pair_{i}_{j}', r_pair))
        if args.mark_lat is not None and args.mark_lon is not None:
            subset_mark = [m for m in baseline_motion if angular_distance(m['cog_deg'], bearing_deg(m['lat'], m['lon'], args.mark_lat, args.mark_lon)) <= args.max_off_angle]
            r = run_estimator_on_motion(subset_mark, min_points=args.min_points) if len(subset_mark) >= args.min_points else None
            if r:
                candidates_debug.append(('mark_latlon', r))
        subset_speed = [m for m in motion_all if m['speed_m_s'] >= args.min_speed and m['speed_m_s'] <= args.beat_max_speed]
        r = run_estimator_on_motion(subset_speed, min_points=args.min_points) if len(subset_speed) >= args.min_points else None
        if r:
            candidates_debug.append(('beat_speed', r))
        for ttag, rr in candidates_debug:
            print(f" - {ttag}: spread={rr['overall_spread_deg']:.1f} deg, bis_prob={rr['bisector_prob']:.3f}, upwind={rr['upwind_likelihood']:.3f}, conf={rr['confidence_0_1']:.3f}, n={rr['n_points_used']}")

    if args.plot:
        try:
            plot_result(subset, res, title_extra=tag)
        except Exception as e:
            print('Plotting failed:', e, file=sys.stderr)

if __name__ == '__main__':
    main()
