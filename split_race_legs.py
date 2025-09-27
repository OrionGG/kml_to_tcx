import xml.etree.ElementTree as ET
import math
from datetime import datetime
import os
import numpy as np
from collections import defaultdict

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing between two points."""
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    d_lon = lon2 - lon1
    
    x = math.cos(lat2) * math.sin(d_lon)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters using Haversine formula."""
    R = 6371000  # Earth's radius in meters
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def estimate_true_wind(trackpoints, namespaces):
    """Estimate true wind direction by analyzing first upwind leg."""
    # First, get the main direction of the first leg (first 3 minutes)
    bearings = []
    start_time = datetime.strptime(trackpoints[0].find(".//{" + namespaces[''] + "}Time").text, "%Y-%m-%dT%H:%M:%SZ")
    
    # Analyze first 3 minutes to get upwind direction
    window_size = 5  # Use a moving window of 5 points for smoother bearings
    current_window = []
    
    for i in range(len(trackpoints)-1):
        time_str = trackpoints[i].find(".//{" + namespaces[''] + "}Time").text
        time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
        
        if (time - start_time).total_seconds() > 180:  # 3 minutes
            break
        
        lat1 = float(trackpoints[i].find(".//{" + namespaces[''] + "}LatitudeDegrees").text)
        lon1 = float(trackpoints[i].find(".//{" + namespaces[''] + "}LongitudeDegrees").text)
        lat2 = float(trackpoints[i+1].find(".//{" + namespaces[''] + "}LatitudeDegrees").text)
        lon2 = float(trackpoints[i+1].find(".//{" + namespaces[''] + "}LongitudeDegrees").text)
        
        bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        current_window.append(bearing)
        
        if len(current_window) >= window_size:
            # Calculate average bearing for the window
            avg_bearing = sum(current_window) / len(current_window)
            bearings.append(avg_bearing)
            current_window.pop(0)
    
    if not bearings:
        raise ValueError("Not enough data points to estimate wind direction")

    # Get the most common bearing range (binned to 10 degree intervals)
    binned_bearings = np.array(bearings) // 10 * 10
    unique, counts = np.unique(binned_bearings, return_counts=True)
    main_bearing = float(unique[np.argmax(counts)])
    
    # In an ILCA (Laser) boat, typical upwind sailing angle is around 45 degrees to true wind
    # So if we're sailing at bearing B, true wind is at (B ± 45) depending on tack
    possible_wind_1 = (main_bearing + 45) % 360
    possible_wind_2 = (main_bearing - 45) % 360
    
    # Look at the first downwind leg (should be after a tack, around 3-5 minutes into the race)
    later_bearings = []
    for i in range(len(trackpoints)-1):
        time_str = trackpoints[i].find(".//{" + namespaces[''] + "}Time").text
        time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
        
        if 180 <= (time - start_time).total_seconds() <= 300:  # 3-5 minutes
            lat1 = float(trackpoints[i].find(".//{" + namespaces[''] + "}LatitudeDegrees").text)
            lon1 = float(trackpoints[i].find(".//{" + namespaces[''] + "}LongitudeDegrees").text)
            lat2 = float(trackpoints[i+1].find(".//{" + namespaces[''] + "}LatitudeDegrees").text)
            lon2 = float(trackpoints[i+1].find(".//{" + namespaces[''] + "}LongitudeDegrees").text)
            bearing = calculate_bearing(lat1, lon1, lat2, lon2)
            later_bearings.append(bearing)
    
    if later_bearings:
        avg_later_bearing = sum(later_bearings) / len(later_bearings)
        # The correct wind direction should result in a TWA of around 45-50 degrees
        twa1 = calculate_twa(avg_later_bearing, possible_wind_1)
        twa2 = calculate_twa(avg_later_bearing, possible_wind_2)
        true_wind_direction = possible_wind_1 if abs(twa1 - 45) < abs(twa2 - 45) else possible_wind_2
    else:
        # If we can't validate with later bearings, use the first estimate
        true_wind_direction = possible_wind_1
    
    return true_wind_direction

def calculate_twa(cog, true_wind_direction):
    """Calculate True Wind Angle from course over ground and true wind direction."""
    twa = abs(cog - true_wind_direction)
    if twa > 180:
        twa = 360 - twa
    return twa

def identify_leg_type(avg_cog, true_wind_direction, leg_count=0):
    """Identify the type of leg based on True Wind Angle (TWA) and leg sequence."""
    twa = calculate_twa(avg_cog, true_wind_direction)
    relative_wind = (avg_cog - true_wind_direction + 360) % 360
    
    # Expected leg sequence for this race:
    # 1. Upwind
    # 2. Reach starboard
    # 3. Downwind
    # 4. Upwind
    # 5. Downwind
    # 6. Reach port
    
    expected_sequence = [
        (0, "upwind", (0, 60)),
        (1, "reach_starboard", (60, 120)),
        (2, "downwind", (120, 180)),
        (3, "upwind", (0, 60)),
        (4, "downwind", (120, 180)),
        (5, "reach_port", (60, 120))
    ]
    
    # Use the expected leg type if we're within sequence
    for seq_num, leg_type, (min_twa, max_twa) in expected_sequence:
        if seq_num == leg_count and min_twa <= twa <= max_twa:
            return leg_type
    
    # Fallback to basic classification if sequence doesn't match
    if twa <= 60:
        return "upwind"
    elif twa >= 120:
        return "downwind"
    else:
        wind_side = (avg_cog - true_wind_direction + 360) % 360
        return "reach_starboard" if wind_side < 180 else "reach_port"

def create_tcx_for_leg(leg_points, leg_number, leg_type, output_dir, namespaces):
    """Create a new TCX file for a specific leg."""
    # Create the root element with all necessary namespaces
    root = ET.Element("TrainingCenterDatabase")
    for prefix, uri in namespaces.items():
        if prefix:
            root.set(f"xmlns:{prefix}", uri)
        else:
            root.set("xmlns", uri)
    
    activities = ET.SubElement(root, "Activities")
    activity = ET.SubElement(activities, "Activity")
    activity.set("Sport", "Other")
    
    # Use the first point's time as the ID
    id_elem = ET.SubElement(activity, "Id")
    id_elem.text = leg_points[0].find(".//{" + namespaces[''] + "}Time").text
    
    # Create lap element
    lap = ET.SubElement(activity, "Lap")
    lap.set("StartTime", leg_points[0].find(".//{" + namespaces[''] + "}Time").text)
    
    # Create track element and add all points
    track = ET.SubElement(lap, "Track")
    for point in leg_points:
        track.append(point)
    
    # Create the output file
    tree = ET.ElementTree(root)
    output_file = os.path.join(output_dir, f"leg_{leg_number}_{leg_type}.tcx")
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    
def split_race_into_legs(tcx_file, output_dir):
    """Split a race TCX file into separate legs."""
    # Parse the TCX file
    tree = ET.parse(tcx_file)
    root = tree.getroot()
    
    # Get all namespaces
    namespaces = {'': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
                  'ns0': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2',
                  'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'}
    
    # Find all trackpoints
    trackpoints = root.findall(".//{" + namespaces[''] + "}Trackpoint")
    
    # First, estimate the true wind direction from the entire track
    true_wind_direction = estimate_true_wind(trackpoints, namespaces)
    print(f"Estimated true wind direction: {true_wind_direction:.1f}°")
    
    current_leg = []
    legs = []
    window_size = 20  # Increased window size for more stable TWA calculation
    bearing_window = []
    
    # Process all trackpoints
    min_leg_duration = 180  # Minimum 3 minutes per leg
    leg_start_time = datetime.strptime(trackpoints[0].find(".//{" + namespaces[''] + "}Time").text, "%Y-%m-%dT%H:%M:%SZ")
    
    for i in range(len(trackpoints)):
        current_leg.append(trackpoints[i])
        
        if i < len(trackpoints) - 1:
            # Calculate bearing to next point
            lat1 = float(trackpoints[i].find(".//{" + namespaces[''] + "}LatitudeDegrees").text)
            lon1 = float(trackpoints[i].find(".//{" + namespaces[''] + "}LongitudeDegrees").text)
            lat2 = float(trackpoints[i + 1].find(".//{" + namespaces[''] + "}LatitudeDegrees").text)
            lon2 = float(trackpoints[i + 1].find(".//{" + namespaces[''] + "}LongitudeDegrees").text)
            
            bearing = calculate_bearing(lat1, lon1, lat2, lon2)
            bearing_window.append(bearing)
            
            time_str = trackpoints[i].find(".//{" + namespaces[''] + "}Time").text
            current_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
            
            # Print first 15 minutes for debugging
            start_time = datetime.strptime(trackpoints[0].find(".//{" + namespaces[''] + "}Time").text, "%Y-%m-%dT%H:%M:%SZ")
            if (current_time - start_time).total_seconds() <= 900:
                twa = calculate_twa(bearing, true_wind_direction)
                print(f"Time: {time_str}, Bearing: {bearing:.1f}°, TWA: {twa:.1f}°")
            
            if len(bearing_window) > window_size:
                bearing_window.pop(0)
            
            # If we have enough points to analyze
            if len(bearing_window) == window_size:
                avg_bearing = sum(bearing_window) / len(bearing_window)
                twa = calculate_twa(avg_bearing, true_wind_direction)
                
                # Only consider splitting if minimum leg duration is met
                leg_duration = (current_time - leg_start_time).total_seconds()
                
                if leg_duration >= min_leg_duration:
                    # Calculate current TWA
                    current_twa = calculate_twa(avg_bearing, true_wind_direction)
                    
                    # Get current leg type
                    current_type = identify_leg_type(avg_bearing, true_wind_direction)
                    
                    # Only split if we're starting a new leg or if there's a significant change in TWA
                    if len(legs) == 0:
                        if len(current_leg) > window_size:
                            legs.append((current_leg.copy(), current_type))
                            current_leg = []
                            leg_start_time = current_time
                    else:
                        prev_leg_type = identify_leg_type(
                            sum(bearing_window[:window_size//2]) / (window_size//2),
                            true_wind_direction
                        )
                        
                        # Split if leg type changes (upwind -> reach -> downwind)
                        if prev_leg_type != current_type and leg_duration > 300:  # Minimum 5 minutes per leg
                            legs.append((current_leg.copy(), current_type))
                            current_leg = []
                            leg_start_time = current_time
    
    # Add the last leg
    if current_leg:
        if bearing_window:
            avg_bearing = sum(bearing_window) / len(bearing_window)
            legs.append((current_leg, identify_leg_type(avg_bearing, true_wind_direction)))
        else:
            legs.append((current_leg, "finish"))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create TCX files for each leg
    for i, (leg_points, leg_type) in enumerate(legs):
        create_tcx_for_leg(leg_points, i + 1, leg_type, output_dir, namespaces)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python split_race_legs.py <tcx_file>")
        sys.exit(1)
    
    tcx_file = sys.argv[1]
    output_dir = os.path.join(os.path.dirname(tcx_file), "race_legs")
    split_race_into_legs(tcx_file, output_dir)
    print(f"Race has been split into legs. Output files are in: {output_dir}")