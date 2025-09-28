#!/usr/bin/env python3
"""
split_tcx_legs.py

Split a single TCX track into 8 legs (TCX files) using waypoint times provided
as times from the beginning of the track. Waypoints may be entered as:
  - mm:ss  (e.g. 03:30 means 3 minutes 30 seconds from start)
  - decimal minutes (e.g. 3.5 means 3.5 minutes = 3 minutes 30 seconds)
  - a comma-separated list or space-separated values

Assumptions & behavior
- The input TCX must contain at least one Activity with Trackpoints (TrainingCenterDatabase schema).
- Waypoints are provided as 7 time values (in mm:ss or decimal minutes) in ascending order.
- The script will create 8 TCX files (one per leg) named
  <input_basename>_leg_01.tcx ... <input_basename>_leg_08.tcx
- Segment boundaries are:
    segment 0: start_of_track .. waypoint1
    segment 1: waypoint1 .. waypoint2
    ...
    segment 6: waypoint6 .. waypoint7
    segment 7: waypoint7 .. end_of_track
- Timestamps are preserved. Lap StartTime is set to the segment start time.
- If a segment has no trackpoints (rare), a small message is printed and an empty
  TCX with just the Lap and no Trackpoints is still written.

Usage
    python split_tcx_legs.py --input /path/to/track.tcx --waypoints 0 3:30 8:12 12:00 17 21:00 24:00

Or provide waypoints as a comma-separated list:
    --waypoints "0,03:30,08:12,12:00,17,21:00,24:00"

Notes
- Each numeric value without a colon is interpreted as minutes (decimal minutes). E.g. `3.5` = 3 minutes 30 seconds.
- Values with format `MM:SS` or `M:SS` are parsed as minutes and seconds.

"""

import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import copy
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Split TCX into 8 legs by waypoint times (mm:ss or decimal minutes)")
    p.add_argument("--input", "-i", required=True, help="Input TCX file path")
    p.add_argument("--waypoints", "-w", required=True, nargs='+',
                   help=("Seven waypoint times. Use mm:ss (e.g. 03:30) or decimal minutes (e.g. 3.5). "
                         "Either provide 7 separate values or a single comma-separated string."))
    p.add_argument("--outdir", "-o", default=None, help="Output directory (defaults to input file folder)")
    return p.parse_args()


def parse_time_token(tok):
    """Parse a single token into seconds (float).

    Accepted formats:
      - "MM:SS" or "M:SS" or "H:MM:SS" -> interpreted as minutes:seconds (or hours:minutes:seconds)
      - decimal number like "3.5" -> interpreted as minutes (3.5 minutes = 210 seconds)
      - integer like "3" -> interpreted as minutes
    """
    tok = tok.strip()
    if not tok:
        raise ValueError('Empty waypoint token')
    if ':' in tok:
        parts = tok.split(':')
        try:
            # allow H:MM:SS or MM:SS
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
        # interpret as decimal minutes
        try:
            minutes = float(tok)
        except ValueError:
            raise ValueError(f'Invalid numeric time value: {tok}')
        total_seconds = minutes * 60.0
    return total_seconds


def _parse_waypoints(wp_args):
    # wp_args can be a list like ['0','3:30',...] or a single string '0,3:30,...'
    if len(wp_args) == 1 and ',' in wp_args[0]:
        items = [x.strip() for x in wp_args[0].split(',') if x.strip()]
    else:
        items = wp_args
    if len(items) != 7:
        raise ValueError(f'Expected 7 waypoint values, got {len(items)}')
    secs = []
    for t in items:
        secs.append(parse_time_token(t))
    # ensure ascending order
    if any(secs[i] > secs[i+1] for i in range(len(secs)-1)):
        raise ValueError('Waypoints must be in non-decreasing order (ascending).')
    return secs


def _iso_to_dt(s):
    # handle trailing Z and subsecond formats by replacing Z with +00:00
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    # Python's fromisoformat handles offsets like +00:00
    return datetime.fromisoformat(s)


def _dt_to_iso_z(dt):
    # return ISO with Z for UTC if tzinfo is UTC-like
    if dt.tzinfo is None:
        return dt.isoformat() + 'Z'
    else:
        offset = dt.utcoffset()
        if offset is not None and offset.total_seconds() == 0:
            return dt.replace(tzinfo=None).isoformat() + 'Z'
        return dt.isoformat()


def collect_trackpoints(root):
    # detect namespace
    tag = root.tag
    ns = ''
    if tag.startswith('{'):
        ns = tag[1:].split('}')[0]
    nsmap = {'ns': ns} if ns else {}

    # find all Trackpoint elements under Activities/Activity/Lap/Track
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

    # sort by timestamp just in case
    trackpoints.sort(key=lambda x: x[0])
    return trackpoints, ns


def build_segment_tcx(original_root, trackpoints_segment, segment_start_dt, segment_end_dt, ns, activity_sport, activity_id_text, out_path):
    # Build minimal TCX tree for the segment
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

    track_el = ET.SubElement(lap, tcx_ns + 'Track')

    # append deep-copied trackpoints
    for ts, tp in trackpoints_segment:
        track_el.append(copy.deepcopy(tp))

    # Optionally include Creator if present in original root
    creator = original_root.find('.//' + tcx_ns + 'Creator')
    if creator is not None:
        activity.append(copy.deepcopy(creator))

    tree = ET.ElementTree(root)
    tree.write(out_path, encoding='utf-8', xml_declaration=True)


def main():
    args = parse_args()
    waypoints_secs = _parse_waypoints(args.waypoints)  # list of seconds from start
    inpath = args.input
    outdir = args.outdir or os.path.dirname(os.path.abspath(inpath))
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

    # build absolute waypoint datetimes from seconds
    wp_dts = [start_time + timedelta(seconds=sec) for sec in waypoints_secs]

    # build sequence of boundaries
    boundaries = [start_time] + wp_dts + [end_time]

    # extract activity attributes for recreation
    activity_el = root.find('.//' + (f'{{{ns}}}Activity' if ns else 'Activity'))
    activity_sport = activity_el.get('Sport') if activity_el is not None and activity_el.get('Sport') else 'Other'
    activity_id_el = activity_el.find(f'{{{ns}}}Id') if activity_el is not None else None
    activity_id_text = activity_id_el.text if activity_id_el is not None and activity_id_el.text else datetime.utcnow().isoformat()

    base = os.path.splitext(os.path.basename(inpath))[0]

    for i in range(8):
        seg_start = boundaries[i]
        seg_end = boundaries[i+1]
        # define inclusion: include points where ts >= seg_start and (ts < seg_end or i==7)
        selected = []
        for ts, tp in trackpoints:
            if ts < seg_start:
                continue
            if i < 7:
                if ts >= seg_end:
                    break
                selected.append((ts, tp))
            else:
                # last segment -> include up to and including end
                if ts <= seg_end:
                    selected.append((ts, tp))
                else:
                    break

        outname = f"{base}_leg_{i+1:02d}.tcx"
        outpath = os.path.join(outdir, outname)
        if not selected:
            print(f'Warning: segment {i+1} has 0 trackpoints. Creating empty TCX with lap timestamps.')
        build_segment_tcx(root, selected, seg_start, seg_end, ns, activity_sport, f"{activity_id_text}_leg_{i+1}", outpath)
        print(f'Wrote {outpath}  ({len(selected)} trackpoints)')

    print('Done.')


if __name__ == '__main__':
    main()
