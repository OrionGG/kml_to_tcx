import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class TrackPoint:
    time: datetime
    latitude: float
    longitude: float
    speed: float  # in meters per second
    course: float  # in degrees

@dataclass
class LegData:
    start_time: datetime
    end_time: datetime
    leg_type: str
    avg_twa: float
    avg_speed: float

def parse_tcx_file(tcx_path: str) -> List[TrackPoint]:
    """Parse TCX file and extract track points with time, position, and speed data"""
    tree = ET.parse(tcx_path)
    root = tree.getroot()
    
    # Define namespace
    ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
          'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'}
    
    track_points = []
    
    # Find all Trackpoint elements
    for trkpt in root.findall('.//ns:Trackpoint', ns):
        time_elem = trkpt.find('ns:Time', ns)
        position = trkpt.find('ns:Position', ns)
        
        if time_elem is not None and position is not None:
            time = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
            lat = float(position.find('ns:LatitudeDegrees', ns).text)
            lon = float(position.find('ns:LongitudeDegrees', ns).text)
            
            # Extract speed and course from extensions if available
            speed = 0.0
            course = 0.0
            extensions = trkpt.find('.//ns3:TPX', ns)
            if extensions is not None:
                speed_elem = extensions.find('ns3:Speed', ns)
                course_elem = extensions.find('ns3:Course', ns)
                if speed_elem is not None:
                    speed = float(speed_elem.text)
                if course_elem is not None:
                    course = float(course_elem.text)
            
            track_points.append(TrackPoint(time, lat, lon, speed, course))
    
    return track_points

def estimate_true_wind(boat_speed: float, apparent_wind_speed: float, 
                      apparent_wind_angle: float) -> Tuple[float, float]:
    """
    Calculate true wind speed and angle from apparent wind and boat speed
    
    Args:
        boat_speed: Boat speed in m/s
        apparent_wind_speed: Apparent wind speed in m/s
        apparent_wind_angle: Apparent wind angle in degrees
    
    Returns:
        Tuple of (true_wind_speed, true_wind_angle) in m/s and degrees
    """
    # Convert angle to radians
    awa_rad = math.radians(apparent_wind_angle)
    
    # Calculate x and y components
    x = apparent_wind_speed * math.sin(awa_rad)
    y = apparent_wind_speed * math.cos(awa_rad) - boat_speed
    
    # Calculate true wind speed and angle
    true_wind_speed = math.sqrt(x*x + y*y)
    true_wind_angle = math.degrees(math.atan2(x, y))
    
    # Normalize angle to 0-360 range
    true_wind_angle = (true_wind_angle + 360) % 360
    
    return true_wind_speed, true_wind_angle

def detect_legs(track_points: List[TrackPoint], min_duration: timedelta = timedelta(minutes=3)) -> List[LegData]:
    """
    Detect sailing legs based on TWA ranges and minimum duration
    
    Args:
        track_points: List of track points with time, position, and speed data
        min_duration: Minimum duration for a leg (default 3 minutes)
    
    Returns:
        List of LegData objects containing leg information
    """
    legs = []
    current_leg_type = None
    leg_start_idx = 0
    
    for i in range(1, len(track_points)):
        # For this example, we'll use the course as TWA (in a real implementation,
        # you would need actual wind data to calculate TWA)
        twa = track_points[i].course
        
        # Determine leg type based on TWA
        if twa < 70:
            leg_type = "upwind"
        elif twa < 120:
            leg_type = "reach"
        else:
            leg_type = "downwind"
        
        # Check if we're starting a new leg
        if leg_type != current_leg_type:
            if current_leg_type is not None:
                # Check if previous leg meets minimum duration
                duration = track_points[i-1].time - track_points[leg_start_idx].time
                if duration >= min_duration:
                    # Calculate leg statistics
                    leg_points = track_points[leg_start_idx:i]
                    avg_twa = sum(tp.course for tp in leg_points) / len(leg_points)
                    avg_speed = sum(tp.speed for tp in leg_points) / len(leg_points)
                    
                    legs.append(LegData(
                        start_time=track_points[leg_start_idx].time,
                        end_time=track_points[i-1].time,
                        leg_type=current_leg_type,
                        avg_twa=avg_twa,
                        avg_speed=avg_speed
                    ))
            
            # Start new leg
            current_leg_type = leg_type
            leg_start_idx = i
    
    # Handle the last leg
    if current_leg_type is not None:
        duration = track_points[-1].time - track_points[leg_start_idx].time
        if duration >= min_duration:
            leg_points = track_points[leg_start_idx:]
            avg_twa = sum(tp.course for tp in leg_points) / len(leg_points)
            avg_speed = sum(tp.speed for tp in leg_points) / len(leg_points)
            
            legs.append(LegData(
                start_time=track_points[leg_start_idx].time,
                end_time=track_points[-1].time,
                leg_type=current_leg_type,
                avg_twa=avg_twa,
                avg_speed=avg_speed
            ))
    
    return legs

def analyze_race_file(tcx_path: str):
    """Analyze a single race file and print leg information"""
    print(f"\nAnalyzing race file: {tcx_path}")
    print("-" * 60)
    
    # Parse TCX file
    track_points = parse_tcx_file(tcx_path)
    if not track_points:
        print("No track points found in file")
        return
    
    # Detect legs
    legs = detect_legs(track_points)
    
    # Print results
    print(f"Found {len(legs)} legs:")
    for i, leg in enumerate(legs, 1):
        duration = (leg.end_time - leg.start_time).total_seconds() / 60
        print(f"\nLeg {i} ({leg.leg_type.upper()}):")
        print(f"  Duration: {duration:.1f} minutes")
        print(f"  Average TWA: {leg.avg_twa:.1f}°")
        print(f"  Average Speed: {leg.avg_speed * 1.94384:.1f} knots")  # Convert m/s to knots
        print(f"  Start Time: {leg.start_time.strftime('%H:%M:%S')}")
        print(f"  End Time: {leg.end_time.strftime('%H:%M:%S')}")

def main():
    """Main program entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_race_legs.py <tcx_file_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        # Process all TCX files in directory
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith('.tcx'):
                    tcx_path = os.path.join(root, file)
                    analyze_race_file(tcx_path)
    else:
        # Process single TCX file
        analyze_race_file(path)

if __name__ == "__main__":
    import os
    main()