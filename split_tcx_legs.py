#!/usr/bin/env python3
"""
split_tcx_legs.py

Split a TCX file into 8 named legs while preserving original TCX structure
and ensuring the ActivityExtension namespace appears with prefix "ns0" in output.

Usage:
    python split_tcx_legs.py -i ./Race1/OGG.tcx -w 5:00 17:34 22:04 31:34 43:44 52:00 53:36
"""
import argparse
import copy
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# Namespace URIs
TCX_NS = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
ACT_EXT_NS = "http://www.garmin.com/xmlschemas/ActivityExtension/v2"

# Register namespaces:
# - default namespace with no prefix
# - temporary prefix 'ae' for ActivityExtension; we'll post-process files to change 'ae' -> 'ns0'
ET.register_namespace('', TCX_NS)
ET.register_namespace('ae', ACT_EXT_NS)

LEG_NAMES = [
    'leg_00_pre-start',
    'leg_01_upwind',
    'leg_02_starboard_reach',
    'leg_03_downwind',
    'leg_04_upwind',
    'leg_05_downwind',
    'leg_06_port_reach',
    'leg_07_post_finish',
]


def parse_args():
    p = argparse.ArgumentParser(description="Split TCX into 8 named legs preserving namespace style")
    p.add_argument('-i', '--input', required=True, help='Input TCX file path')
    p.add_argument('-w', '--waypoints', required=True, nargs='+', help='Seven waypoint times (mm:ss or decimal minutes)')
    p.add_argument('-o', '--outdir', default=None, help='Base output dir (defaults to input file dir)')
    return p.parse_args()


def parse_time_token(tok: str) -> int:
    tok = tok.strip()
    if not tok:
        raise ValueError("Empty token")
    if ':' in tok:
        parts = tok.split(':')
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
        elif len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        else:
            raise ValueError(f"Bad time format: {tok}")
    else:
        # decimal minutes -> seconds
        try:
            minutes = float(tok)
        except ValueError:
            raise ValueError(f"Bad numeric time: {tok}")
        return int(round(minutes * 60.0))


def parse_waypoints(wp_args):
    if len(wp_args) == 1 and ',' in wp_args[0]:
        items = [x.strip() for x in wp_args[0].split(',') if x.strip()]
    else:
        items = wp_args
    if len(items) != 7:
        raise ValueError(f"Expected 7 waypoint values, got {len(items)}")
    secs = [parse_time_token(x) for x in items]
    if any(secs[i] > secs[i + 1] for i in range(len(secs) - 1)):
        raise ValueError("Waypoints must be in ascending order")
    return secs


def iso_to_dt(s: str) -> datetime:
    if s is None:
        raise ValueError("Empty time string")
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    return datetime.fromisoformat(s)


def dt_to_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        return dt.isoformat() + 'Z'
    else:
        off = dt.utcoffset()
        if off is not None and off.total_seconds() == 0:
            return dt.replace(tzinfo=None).isoformat() + 'Z'
        return dt.isoformat()


def haversine_m(lat1, lon1, lat2, lon2):
    # returns meters
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def collect_trackpoints(tree):
    root = tree.getroot()
    tcx_ns = f'{{{TCX_NS}}}'
    tps = root.findall('.//' + tcx_ns + 'Trackpoint')
    return tps, root


def compute_segment_metrics(trackpoints, tcx_ns_braced):
    if not trackpoints:
        return 0.0, 0.0, 0.0
    times = []
    positions = []
    ext_speeds = []
    for tp in trackpoints:
        t_el = tp.find(tcx_ns_braced + 'Time')
        if t_el is None or not t_el.text:
            continue
        try:
            times.append(iso_to_dt(t_el.text))
        except Exception:
            continue
        pos = tp.find(tcx_ns_braced + 'Position')
        if pos is not None:
            lat_el = pos.find(tcx_ns_braced + 'LatitudeDegrees')
            lon_el = pos.find(tcx_ns_braced + 'LongitudeDegrees')
            if lat_el is not None and lon_el is not None and lat_el.text and lon_el.text:
                try:
                    positions.append((float(lat_el.text), float(lon_el.text)))
                except Exception:
                    positions.append(None)
            else:
                positions.append(None)
        else:
            positions.append(None)

        # look for ae:Speed inside Extensions -> will serialize using 'ae' prefix
        ext = tp.find(tcx_ns_braced + 'Extensions')
        if ext is not None:
            for child in ext:
                for sp in child.findall('.//' + f'{{{ACT_EXT_NS}}}Speed'):
                    try:
                        ext_speeds.append(float(sp.text))
                    except Exception:
                        pass

    if not times:
        return 0.0, 0.0, 0.0
    total_seconds = (times[-1] - times[0]).total_seconds()

    # distance sum for adjacent positions
    dist_m = 0.0
    prev_pos = None
    prev_time = None
    computed_speeds = []
    for idx, pos in enumerate(positions):
        if pos is None:
            continue
        if prev_pos is not None and prev_time is not None:
            try:
                cur_time = times[idx]
            except Exception:
                cur_time = None
            seg = haversine_m(prev_pos[0], prev_pos[1], pos[0], pos[1])
            dist_m += seg
            if cur_time is not None:
                dt = (cur_time - prev_time).total_seconds()
                if dt > 0:
                    computed_speeds.append(seg / dt)
        # update
        try:
            prev_pos = pos
            prev_time = times[idx]
        except Exception:
            prev_pos = pos
            prev_time = None

    # determine max speed
    max_speed = 0.0
    if ext_speeds:
        try:
            max_speed = max(ext_speeds)
        except Exception:
            max_speed = 0.0
    elif computed_speeds:
        max_speed = max(computed_speeds)
    else:
        max_speed = 0.0

    return float(total_seconds), float(dist_m), float(max_speed)


def build_segment_file_preserve(trackpoints_segment, activity_sport, activity_id_text, out_path):
    # create root using default TCX namespace (no prefix)
    tcx_ns_braced = f'{{{TCX_NS}}}'
    root = ET.Element(tcx_ns_braced + 'TrainingCenterDatabase')

    activities = ET.SubElement(root, tcx_ns_braced + 'Activities')
    activity = ET.SubElement(activities, tcx_ns_braced + 'Activity', {'Sport': activity_sport})
    id_el = ET.SubElement(activity, tcx_ns_braced + 'Id')
    id_el.text = activity_id_text

    # Lap start from first TP time (string)
    if trackpoints_segment:
        first_tp = trackpoints_segment[0]
        t_el = first_tp.find(tcx_ns_braced + 'Time')
        lap_start = t_el.text if (t_el is not None and t_el.text) else dt_to_iso_z(datetime.utcnow())
    else:
        lap_start = dt_to_iso_z(datetime.utcnow())

    lap = ET.SubElement(activity, tcx_ns_braced + 'Lap', {'StartTime': lap_start})

    # compute metrics and write them
    total_time_s, dist_m, max_speed = compute_segment_metrics(trackpoints_segment, tcx_ns_braced)
    ET.SubElement(lap, tcx_ns_braced + 'TotalTimeSeconds').text = f"{total_time_s:.1f}"
    ET.SubElement(lap, tcx_ns_braced + 'DistanceMeters').text = f"{dist_m:.6f}"
    ET.SubElement(lap, tcx_ns_braced + 'MaximumSpeed').text = f"{max_speed:.2f}"

    # Track and deep-copy original Trackpoints (preserving Extensions etc.)
    track_el = ET.SubElement(lap, tcx_ns_braced + 'Track')
    for tp in trackpoints_segment:
        track_el.append(copy.deepcopy(tp))

    # Creator (minimal)
    creator = ET.SubElement(activity, tcx_ns_braced + 'Creator')
    name = ET.SubElement(creator, tcx_ns_braced + 'Name')
    name.text = 'split_tcx_legs'

    # write file with UTF-8 declaration
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass
    tree.write(out_path, encoding='UTF-8', xml_declaration=True)

    # Post-process the file to replace 'ae' prefix with 'ns0' so it matches your original style
    # Replace xmlns:ae= -> xmlns:ns0= and <ae: -> <ns0: and </ae: -> </ns0:
    # This is a safe textual swap because 'ae' was chosen solely for this namespace.
    with open(out_path, 'r', encoding='UTF-8') as f:
        txt = f.read()
    txt = txt.replace('xmlns:ae="', 'xmlns:ns0="')
    txt = txt.replace('<ae:', '<ns0:')
    txt = txt.replace('</ae:', '</ns0:')
    # Also handle the case where ElementTree uses qualified names in attributes (rare)
    txt = txt.replace(' ae:', ' ns0:')
    with open(out_path, 'w', encoding='UTF-8') as f:
        f.write(txt)


def main():
    args = parse_args()
    waypoint_secs = parse_waypoints(args.waypoints)

    inpath = args.input
    outbase = args.outdir or os.path.dirname(os.path.abspath(inpath))
    outdir = os.path.join(outbase, 'race_legs')
    os.makedirs(outdir, exist_ok=True)

    tree = ET.parse(inpath)
    tps, orig_root = collect_trackpoints(tree)
    if not tps:
        print("No Trackpoints found in input TCX")
        return

    tcx_ns_braced = f'{{{TCX_NS}}}'
    first_time = iso_to_dt(tps[0].find(tcx_ns_braced + 'Time').text)
    last_time = iso_to_dt(tps[-1].find(tcx_ns_braced + 'Time').text)

    wp_dts = [first_time + timedelta(seconds=s) for s in waypoint_secs]
    boundaries = [first_time] + wp_dts + [last_time]

    activity_el = orig_root.find('.//' + tcx_ns_braced + 'Activity')
    activity_sport = activity_el.get('Sport') if activity_el is not None and activity_el.get('Sport') else 'Other'
    base = os.path.splitext(os.path.basename(inpath))[0]

    for i in range(8):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        seg_tps = []
        for tp in tps:
            t_el = tp.find(tcx_ns_braced + 'Time')
            if t_el is None or not t_el.text:
                continue
            try:
                ts = iso_to_dt(t_el.text)
            except Exception:
                continue
            if ts < seg_start:
                continue
            if i < 7:
                if ts >= seg_end:
                    break
                seg_tps.append(tp)
            else:
                if ts <= seg_end:
                    seg_tps.append(tp)
                else:
                    break

        if not seg_tps:
            # duplicate nearest and set its Time to seg_start
            def time_diff(tp):
                t = tp.find(tcx_ns_braced + 'Time')
                if t is None or not t.text:
                    return float('inf')
                return abs((iso_to_dt(t.text) - seg_start).total_seconds())

            nearest = min(tps, key=time_diff)
            dup = copy.deepcopy(nearest)
            t_el = dup.find(tcx_ns_braced + 'Time')
            if t_el is None:
                t_el = ET.SubElement(dup, tcx_ns_braced + 'Time')
            t_el.text = dt_to_iso_z(seg_start)
            seg_tps = [dup]
            print(f"Info: segment {i} ({LEG_NAMES[i]}) empty — duplicated nearest TP at {seg_start.isoformat()}")

        activity_id_text = seg_tps[0].find(tcx_ns_braced + 'Time').text
        outname = f"{base}_{LEG_NAMES[i]}.tcx"
        outpath = os.path.join(outdir, outname)

        build_segment_file_preserve(seg_tps, activity_sport, activity_id_text, outpath)
        print(f"Wrote {outpath} ({len(seg_tps)} trackpoints)")

    print("Done. All legs are in", outdir)


if __name__ == "__main__":
    main()
