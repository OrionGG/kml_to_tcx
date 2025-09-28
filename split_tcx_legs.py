#!/usr/bin/env python3
"""
split_tcx_legs.py

Improved TCX splitter for sailing race legs with Garmin-friendly output.

Features / fixes applied:
- Waypoints accepted in mm:ss or decimal minutes (unchanged)
- Outputs saved into a `race_legs` folder inside the input folder (or inside --outdir)
- Filenames use the requested leg names (leg_00_pre-start ... leg_07_post_finish)
- Garmin-required fields added to each <Lap>: DistanceMeters, Calories, Intensity, TriggerMethod
- Ensures every leg has at least one Trackpoint. If a leg would be empty, the script
  duplicates the nearest available trackpoint and sets its <Time> to the leg start time.
- If Trackpoint elements lack Position or DistanceMeters, we add a DistanceMeters=0.0 to
  make files more acceptable to Garmin Connect.

Usage
    python split_tcx_legs.py -i /path/to/OGG.tcx -w 0 03:30 08:12 12:00 17 21:00 24:00

"""

import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import copy
import os
import sys

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
    p = argparse.ArgumentParser(description="Split TCX into 8 named legs by waypoint times (mm:ss or decimal minutes)")
    p.add_argument("--input", "-i", required=True, help="Input TCX file path")
    p.add_argument("--waypoints", "-w", required=True, nargs='+',
                   help=("Seven waypoint times. Use mm:ss (e.g. 03:30) or decimal minutes (e.g. 3.5). "
                         "Either provide 7 separate values or a single comma-separated string."))
    p.add_argument("--outdir", "-o", default=None, help="Base output directory (defaults to input file folder). "
                   "A subfolder 'race_legs' will be created inside it.")
    return p.parse_args()


def parse_time_token(tok):
    tok = tok.strip()
    if not tok:
        raise ValueError('Empty waypoint token')
    if ':' in tok:
        parts = tok.split(':')
        try:
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                total_seconds = minutes * 60.0 + seconds
            elif len(parts) == 3:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                total_seconds = hours * 3600.0 + minutes * 60.0 + seconds
            else:
                raise ValueError(f'Unsupported time format: {tok}')
        except ValueError:
            raise ValueError(f'Invalid mm:ss or hh:mm:ss value: {tok}')
    else:
        try:
            minutes = float(tok)
        except ValueError:
            raise ValueError(f'Invalid numeric time value: {tok}')
        total_seconds = minutes * 60.0
    return total_seconds


def _parse_waypoints(wp_args):
    if len(wp_args) == 1 and ',' in wp_args[0]:
        items = [x.strip() for x in wp_args[0].split(',') if x.strip()]
    else:
        items = wp_args
    if len(items) != 7:
        raise ValueError(f'Expected 7 waypoint values, got {len(items)}')
    secs = []
    for t in items:
        secs.append(parse_time_token(t))
    if any(secs[i] > secs[i+1] for i in range(len(secs)-1)):
        raise ValueError('Waypoints must be in non-decreasing order (ascending).')
    return secs


def _iso_to_dt(s):
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    return datetime.fromisoformat(s)


def _dt_to_iso_z(dt):
    # Return ISO8601 with Z for UTC-like datetimes, otherwise isoformat()
    if dt.tzinfo is None:
        return dt.isoformat() + 'Z'
    else:
        offset = dt.utcoffset()
        if offset is not None and offset.total_seconds() == 0:
            return dt.replace(tzinfo=None).isoformat() + 'Z'
        return dt.isoformat()


def collect_trackpoints(root):
    tag = root.tag
    ns = ''
    if tag.startswith('{'):
        ns = tag[1:].split('}')[0]

    trackpoints = []
    tp_tag = f'{{{ns}}}Trackpoint' if ns else 'Trackpoint'
    time_tag = f'{{{ns}}}Time' if ns else 'Time'

    for tp in root.findall('.//' + tp_tag):
        t_elem = tp.find(time_tag)
        if t_elem is None or not t_elem.text:
            continue
        try:
            ts = _iso_to_dt(t_elem.text.strip())
        except Exception as e:
            print(f'Warning: unable to parse timestamp "{t_elem.text}" -> {e}', file=sys.stderr)
            continue
        trackpoints.append((ts, tp))

    trackpoints.sort(key=lambda x: x[0])
    return trackpoints, ns


def find_nearest_trackpoint(trackpoints, ref_dt):
    # find the trackpoint with minimal absolute time difference to ref_dt
    best = None
    best_dt = None
    best_diff = None
    for ts, tp in trackpoints:
        diff = abs((ts - ref_dt).total_seconds())
        if best is None or diff < best_diff:
            best = (ts, tp)
            best_dt = ts
            best_diff = diff
    return best  # (ts, tp) or None


def ensure_trackpoint_for_time(tp_tuple, new_time_dt, ns):
    # return a deep-copied trackpoint element, with Time set to new_time_dt and
    # ensure it has either Position or DistanceMeters node (Garmin-friendly)
    _, tp = tp_tuple
    new_tp = copy.deepcopy(tp)
    tcx_ns = f'{{{ns}}}' if ns else ''
    time_tag = tcx_ns + 'Time'
    # replace or create Time element
    t_elem = new_tp.find(time_tag)
    if t_elem is None:
        t_elem = ET.SubElement(new_tp, tcx_ns + 'Time')
    t_elem.text = _dt_to_iso_z(new_time_dt)

    # ensure there is a Position or DistanceMeters
    pos_tag = tcx_ns + 'Position'
    dist_tag = tcx_ns + 'DistanceMeters'
    has_position = new_tp.find(pos_tag) is not None
    has_distance = new_tp.find(dist_tag) is not None
    if not has_position and not has_distance:
        d = ET.SubElement(new_tp, tcx_ns + 'DistanceMeters')
        d.text = '0.0'
    return new_tp


def compute_segment_distance_meters(trackpoints_segment, ns):
    # If trackpoints contain DistanceMeters elements, use last-first. Otherwise 0.0
    if not trackpoints_segment:
        return 0.0
    dist_tag = f'{{{ns}}}DistanceMeters' if ns else 'DistanceMeters'
    first_dm = None
    last_dm = None
    for ts, tp in trackpoints_segment:
        dm_elem = tp.find(dist_tag)
        if dm_elem is not None and dm_elem.text:
            try:
                val = float(dm_elem.text)
            except ValueError:
                continue
            if first_dm is None:
                first_dm = val
            last_dm = val
    if first_dm is not None and last_dm is not None:
        seg_dist = max(0.0, last_dm - first_dm)
        return seg_dist
    return 0.0


def build_segment_tcx(original_root, trackpoints_segment, segment_start_dt, segment_end_dt, ns, activity_sport, activity_id_text, out_path):
    NS = ns
    if NS:
        ET.register_namespace('', NS)
    tcx_ns = f'{{{NS}}}' if NS else ''

    root = ET.Element(tcx_ns + 'TrainingCenterDatabase')
    activities = ET.SubElement(root, tcx_ns + 'Activities')
    activity = ET.SubElement(activities, tcx_ns + 'Activity', {'Sport': activity_sport})
    id_el = ET.SubElement(activity, tcx_ns + 'Id')
    id_el.text = activity_id_text

    lap = ET.SubElement(activity, tcx_ns + 'Lap', {'StartTime': _dt_to_iso_z(segment_start_dt)})
    total_seconds = (segment_end_dt - segment_start_dt).total_seconds()
    ts_el = ET.SubElement(lap, tcx_ns + 'TotalTimeSeconds')
    ts_el.text = f"{total_seconds:.1f}"

    # add required Garmin-friendly fields
    dist_m = compute_segment_distance_meters(trackpoints_segment, ns)
    dist_el = ET.SubElement(lap, tcx_ns + 'DistanceMeters')
    dist_el.text = f"{dist_m:.1f}"

    cal_el = ET.SubElement(lap, tcx_ns + 'Calories')
    cal_el.text = '0'

    int_el = ET.SubElement(lap, tcx_ns + 'Intensity')
    int_el.text = 'Active'

    trig_el = ET.SubElement(lap, tcx_ns + 'TriggerMethod')
    trig_el.text = 'Manual'

    track_el = ET.SubElement(lap, tcx_ns + 'Track')

    # append deep-copied trackpoints
    for ts, tp in trackpoints_segment:
        # ensure timestamp formatting on each TP
        new_tp = copy.deepcopy(tp)
        time_tag = tcx_ns + 'Time'
        t_elem = new_tp.find(time_tag)
        if t_elem is not None and t_elem.text:
            # normalize formatting
            try:
                parsed = _iso_to_dt(t_elem.text.strip())
                t_elem.text = _dt_to_iso_z(parsed)
            except Exception:
                t_elem.text = _dt_to_iso_z(ts)
        else:
            # create time element
            t_elem = ET.SubElement(new_tp, tcx_ns + 'Time')
            t_elem.text = _dt_to_iso_z(ts)
        # ensure position/distance exists
        pos_tag = tcx_ns + 'Position'
        dist_tag = tcx_ns + 'DistanceMeters'
        if new_tp.find(pos_tag) is None and new_tp.find(dist_tag) is None:
            d = ET.SubElement(new_tp, tcx_ns + 'DistanceMeters')
            d.text = '0.0'
        track_el.append(new_tp)

    tree = ET.ElementTree(root)
    tree.write(out_path, encoding='utf-8', xml_declaration=True)


def main():
    args = parse_args()
    waypoints_secs = _parse_waypoints(args.waypoints)  # list of seconds from start
    inpath = args.input
    base_input_dir = args.outdir or os.path.dirname(os.path.abspath(inpath))

    outdir = os.path.join(base_input_dir, 'race_legs')
    os.makedirs(outdir, exist_ok=True)

    try:
        tree = ET.parse(inpath)
    except Exception as e:
        print(f'Failed to parse {inpath}: {e}', file=sys.stderr)
        sys.exit(2)

    root = tree.getroot()
    trackpoints, ns = collect_trackpoints(root)
    if not trackpoints:
        print('No trackpoints found in input TCX.', file=sys.stderr)
        sys.exit(3)

    start_time = trackpoints[0][0]
    end_time = trackpoints[-1][0]

    wp_dts = [start_time + timedelta(seconds=sec) for sec in waypoints_secs]
    boundaries = [start_time] + wp_dts + [end_time]

    activity_el = root.find('.//' + (f'{{{ns}}}Activity' if ns else 'Activity'))
    activity_sport = activity_el.get('Sport') if activity_el is not None and activity_el.get('Sport') else 'Other'
    activity_id_el = activity_el.find(f'{{{ns}}}Id') if activity_el is not None else None
    activity_id_text = activity_id_el.text if activity_id_el is not None and activity_id_el.text else datetime.utcnow().isoformat()

    base = os.path.splitext(os.path.basename(inpath))[0]

    for i in range(8):
        seg_start = boundaries[i]
        seg_end = boundaries[i+1]
        selected = []
        for ts, tp in trackpoints:
            if ts < seg_start:
                continue
            if i < 7:
                if ts >= seg_end:
                    break
                selected.append((ts, tp))
            else:
                if ts <= seg_end:
                    selected.append((ts, tp))
                else:
                    break

        # if no points found for this segment, duplicate the nearest trackpoint and set its time to seg_start
        if not selected:
            nearest = find_nearest_trackpoint(trackpoints, seg_start)
            if nearest is not None:
                new_tp_elem = ensure_trackpoint_for_time(nearest, seg_start, ns)
                # create a tuple (ts, tp) using seg_start as ts
                selected = [(seg_start, new_tp_elem)]
                print(f'Info: segment {i} ({LEG_NAMES[i]}) had no points — duplicated nearest point at {seg_start.isoformat()}')
            else:
                print(f'Warning: no trackpoints available at all to duplicate for segment {i} ({LEG_NAMES[i]}). Skipping.', file=sys.stderr)
                continue

        leg_name = LEG_NAMES[i]
        outname = f"{base}_{leg_name}.tcx"
        outpath = os.path.join(outdir, outname)

        build_segment_tcx(root, selected, seg_start, seg_end, ns, activity_sport, f"{activity_id_text}_{leg_name}", outpath)
        print(f'Wrote {outpath}  ({len(selected)} trackpoints)')

    print(f"All done. Legs written to: {outdir}")


if __name__ == '__main__':
    main()
