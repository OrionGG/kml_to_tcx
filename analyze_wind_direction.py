import xml.etree.ElementTree as ET
import math
from datetime import datetime
import numpy as np
from collections import defaultdict

def read_tcx_file(file_path):
    """Read TCX file and extract trackpoints with time, position, and speed."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Define namespaces
    namespaces = {
        'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
        'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'
    }
    
    trackpoints = []
    
    # Extract all trackpoints
    for trkpt in root.findall('.//ns:Trackpoint', namespaces):
        time = trkpt.find('ns:Time', namespaces).text
        position = trkpt.find('ns:Position', namespaces)
        if position is not None:
            lat = float(position.find('ns:LatitudeDegrees', namespaces).text)
            lon = float(position.find('ns:LongitudeDegrees', namespaces).text)
            
            # Extract speed
            speed_elem = trkpt.find('.//ns3:Speed', namespaces)
            speed = float(speed_elem.text) if speed_elem is not None else 0.0
            
            trackpoints.append({
                'time': datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ'),
                'lat': lat,
                'lon': lon,
                'speed': speed
            })
    
    return trackpoints

def calculate_heading(lat1, lon1, lat2, lon2):
    """Calculate the heading between two points in degrees."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    d_lon = lon2 - lon1
    
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    
    heading = math.degrees(math.atan2(y, x))
    return (heading + 360) % 360

def analyze_wind_direction(trackpoints, min_speed=1.0):
    """
    Analyze track points to estimate wind direction.
    Uses the assumption that ILCA boats are typically fastest on beam reach
    and slowest going directly upwind or downwind.
    """
    if len(trackpoints) < 2:
        return None
    
    # Calculate headings and store with corresponding speeds
    heading_speeds = []
    
    for i in range(len(trackpoints) - 1):
        if trackpoints[i]['speed'] < min_speed:
            continue
            
        heading = calculate_heading(
            trackpoints[i]['lat'],
            trackpoints[i]['lon'],
            trackpoints[i + 1]['lat'],
            trackpoints[i + 1]['lon']
        )
        
        speed = trackpoints[i]['speed']
        heading_speeds.append((heading, speed))
    
    if not heading_speeds:
        return None
    
    # Group speeds by heading sectors (every 10 degrees)
    sectors = defaultdict(list)
    for heading, speed in heading_speeds:
        sector = int(heading / 10) * 10
        sectors[sector].append(speed)
    
    # Calculate average speed for each sector
    avg_speeds = {sector: np.mean(speeds) for sector, speeds in sectors.items()}
    
    if not avg_speeds:
        return None
    
    # Find the sectors with highest speeds (likely beam reach)
    sorted_sectors = sorted(avg_speeds.items(), key=lambda x: x[1], reverse=True)
    
    # The wind direction is typically perpendicular to the fastest point of sail
    # Take the two highest speed sectors and calculate the average
    if len(sorted_sectors) >= 2:
        fast_sectors = sorted_sectors[:2]
        # Calculate the average of the two fastest sectors
        wind_direction = (fast_sectors[0][0] + fast_sectors[1][0]) / 2
        # Add 90 degrees to get wind direction (since fastest point is beam reach)
        wind_direction = (wind_direction + 90) % 360
        return wind_direction
    
    return None

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_wind_direction.py <tcx_file>")
        sys.exit(1)
        
    tcx_file = sys.argv[1]
    try:
        trackpoints = read_tcx_file(tcx_file)
        wind_direction = analyze_wind_direction(trackpoints)
        
        if wind_direction is not None:
            print(f"\nEstimated wind direction: {wind_direction:.1f}°")
            print("Wind direction cardinal: ", end="")
            
            # Convert to cardinal direction
            dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                   'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            ix = round(wind_direction / (360 / len(dirs))) % len(dirs)
            print(f"{dirs[ix]} ({wind_direction:.1f}°)")
        else:
            print("Could not estimate wind direction from the provided data.")
            
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    main()