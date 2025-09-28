#!/usr/bin/env python3
"""
compute_vmg_tcx.py - verbose, robust VMG writer for TCX files.

Usage examples (PowerShell):
  python .\\compute_vmg_tcx.py ".\\Race1\\race_legs\\OGG_leg_01_upwind.tcx" --verbose
  python .\\compute_vmg_tcx.py ".\\Race1\\race_legs\\OGG_leg_01_upwind.tcx" -o ".\\Race1\\race_legs\\OGG_leg_01_upwind vmg.tcx" --units knots --verbose

This writes VMG into <AltitudeMeters> for each <Trackpoint>.
"""
import argparse
import xml.etree.ElementTree as ET
from math import radians, degrees, sin, cos, atan2, sqrt
from datetime import datetime
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Compute VMG from a TCX and write it into AltitudeMeters")
    p.add_argument('input', nargs='?', help='Input TCX file path')
    p.add_argument('-i', '--input-file', dest='input_file', help='Input TCX file path (alternative)')
    p.add_argument('-o', '--output', help='Output TCX file path (default: add \" vmg\" before extension)')
    p.add_argument('-u', '--units', choices=['m/s', 'knots'], default='m/s', help="Units to store in AltitudeMeters")
    p.add_argument('--wind-method', choices=['first-last', 'custom-deg'], default='first-last', help='How to determine wind direction')
    p.add_argument('--wind-deg', type=float, help='Wind bearing in degrees from North (used if wind-method=custom-deg)')
    p.add_argument('--precision', type=int, default=3, help='Decimal places for AltitudeMeters')
    p.add_argument('--verbose', action='store_true', help='Print progress messages')
    return p.parse_args()

# geodesy helpers
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2.0)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2.0)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dlambda = radians(lon2 - lon1)
    x = sin(dlambda) * cos(phi2)
    y = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlambda)
    br = atan2(x, y)
    return (degrees(br) + 360) % 360

def angle_diff_deg(a, b):
    d = (a - b + 180) % 360 - 180
    return d

def parse_iso_time(s):
    if s is None:
        raise ValueError("Time string is None")
    s = s.strip()
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    return datetime.fromisoformat(s)

# TCX load
def load_trackpoints(tcx_path, verbose=False):
    if verbose: print(f"Loading TCX: {tcx_path}")
    tree = ET.parse(tcx_path)
    root = tree.getroot()
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0].strip('{')
    nsmap = {'ns': ns} if ns else {}
    if ns:
        tp_elems = root.findall('.//ns:Trackpoint', nsmap)
    else:
        tp_elems = root.findall('.//Trackpoint')
    points = []
    for tp in tp_elems:
        time_el = tp.find('ns:Time', nsmap) if ns else tp.find('Time')
        pos_el = tp.find('ns:Position', nsmap) if ns else tp.find('Position')
        if time_el is None or pos_el is None:
            continue
        lat_el = pos_el.find('ns:LatitudeDegrees', nsmap) if ns else pos_el.find('LatitudeDegrees')
        lon_el = pos_el.find('ns:LongitudeDegrees', nsmap) if ns else pos_el.find('LongitudeDegrees')
        if lat_el is None or lon_el is None:
            continue
        try:
            t = parse_iso_time(time_el.text)
            lat = float(lat_el.text)
            lon = float(lon_el.text)
        except Exception:
            continue
        points.append({'elem': tp, 'time': t, 'lat': lat, 'lon': lon})
    if verbose: print(f"Found {len(points)} valid trackpoints")
    return tree, root, ns, points

def write_vmg_to_tcx(tree, root, ns, points, vmg_list, out_path, precision=3, verbose=False):
    if verbose: print(f"Main namespace: {ns}")
    
    # Register the main namespace
    if ns:
        ET.register_namespace('', ns)
        nsmap = {'ns': ns}
    else:
        nsmap = {}
    
    # Set up namespace map including ns0 for ActivityExtension
    nsmap['ns0'] = 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'
    
    for i, p in enumerate(points):
        tp = p['elem']
        
        # Remove any existing AltitudeMeters elements
        alt_elements = tp.findall('ns:AltitudeMeters', nsmap) if ns else tp.findall('AltitudeMeters')
        for alt_elem in alt_elements:
            tp.remove(alt_elem)
        
        # Find Extensions -> ns0:TPX -> ns0:Speed path
        extensions = tp.find('ns:Extensions', nsmap) if ns else tp.find('Extensions')
        if extensions is not None:
            tpx = extensions.find('ns0:TPX', nsmap)
            if tpx is not None:
                speed_elem = tpx.find('ns0:Speed', nsmap)
                if speed_elem is not None:
                    # Update the speed value with VMG
                    if i < len(vmg_list):
                        v = vmg_list[i]
                    else:
                        v = vmg_list[-1] if vmg_list else 0.0
                    
                    fmt = f"{{:.{precision}f}}"
                    speed_elem.text = fmt.format(v)
                    
                    if verbose: print(f"Updated trackpoint {i} speed to: {speed_elem.text}")
    
    if verbose: print(f"Writing new TCX to: {out_path}")
    tree.write(out_path, encoding='utf-8', xml_declaration=True)

# VMG calculations
def compute_wind_bearing(points, method='first-last', custom_deg=None):
    if method == 'first-last':
        if len(points) < 2:
            raise ValueError('Need at least 2 points to compute first-last bearing')
        return bearing_deg(points[0]['lat'], points[0]['lon'], points[-1]['lat'], points[-1]['lon'])
    elif method == 'custom-deg':
        if custom_deg is None:
            raise ValueError('custom-deg method requires a numeric wind-deg')
        return custom_deg % 360.0
    else:
        raise ValueError('Unknown wind method')

def compute_vmg_list(points, wind_bearing):
    vmg_list = []
    speeds = []
    bearings = []
    for i in range(len(points)-1):
        p0 = points[i]
        p1 = points[i+1]
        dt = (p1['time'] - p0['time']).total_seconds()
        if dt <= 0:
            dist = 0.0
            speed = 0.0
            br = None
        else:
            dist = haversine_m(p0['lat'], p0['lon'], p1['lat'], p1['lon'])
            speed = dist / dt
            br = bearing_deg(p0['lat'], p0['lon'], p1['lat'], p1['lon'])
        if br is None:
            vmg = 0.0
        else:
            delta_deg = angle_diff_deg(wind_bearing, br)
            delta = radians(delta_deg)
            vmg = speed * cos(delta)
        vmg_list.append(vmg)
        speeds.append(speed)
        bearings.append(br if br is not None else float('nan'))
    return vmg_list, speeds, bearings

def main():
    args = parse_args()
    input_path = args.input or args.input_file
    if not input_path:
        print('Error: input file required (positional or -i/--input-file)', file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(input_path):
        print(f'Error: input file does not exist: {input_path}', file=sys.stderr)
        sys.exit(2)
    out_path = args.output or (os.path.splitext(input_path)[0] + ' vmg' + os.path.splitext(input_path)[1])
    tree, root, ns, points = load_trackpoints(input_path, verbose=args.verbose)
    if len(points) < 2:
        print('Not enough valid trackpoints with position/time to compute VMG (need >=2).', file=sys.stderr)
        sys.exit(1)
    wind_bearing = compute_wind_bearing(points, method=args.wind_method, custom_deg=args.wind_deg)
    if args.verbose: print(f'Wind bearing (deg from North) = {wind_bearing:.2f}')
    vmg_list, speeds, bearings = compute_vmg_list(points, wind_bearing)
    if args.units == 'knots':
        MS_TO_KNOTS = 1.9438444924406046
        vmg_store = [v * MS_TO_KNOTS for v in vmg_list]
    else:
        vmg_store = vmg_list
    write_vmg_to_tcx(tree, root, ns, points, vmg_store, out_path, precision=args.precision, verbose=args.verbose)
    # summary
    import statistics
    avg_vmg = statistics.mean(vmg_list) if vmg_list else 0.0
    max_vmg = max(vmg_list) if vmg_list else 0.0
    min_vmg = min(vmg_list) if vmg_list else 0.0
    avg_speed_ms = statistics.mean(speeds) if speeds else 0.0
    print('')
    print('Summary:')
    print(f'  Track points processed: {len(points)}')
    print(f'  Segments: {len(vmg_list)}')
    print(f'  Wind bearing used: {wind_bearing:.2f}°')
    if args.units == 'knots':
        print(f'  Average VMG written: {statistics.mean(vmg_store):.3f} knots')
    else:
        print(f'  Average VMG written: {avg_vmg:.3f} m/s')
        print(f'  Max VMG: {max_vmg:.3f} m/s')
        print(f'  Min VMG: {min_vmg:.3f} m/s')
        print(f'  Average speed over ground: {avg_speed_ms:.3f} m/s')
    print(f'  New TCX written to: {out_path}')

if __name__ == '__main__':
    main()
