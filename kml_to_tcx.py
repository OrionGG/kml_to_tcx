import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import sys
import os
from math import radians, sin, cos, sqrt, atan2

def create_tcx_header():
    """Create the TCX header with required namespaces"""
    root = ET.Element("TrainingCenterDatabase")
    root.set("xmlns", "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2")
    root.set("xmlns:ns3", "http://www.garmin.com/xmlschemas/ActivityExtension/v2")
    return root

def parse_coordinates(coord_str):
    """Parse KML coordinate string into (lon, lat, alt)"""
    parts = coord_str.strip().split(',')
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0
        except (ValueError, IndexError):
            return None
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters"""
    R = 6371000  # Earth's radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def convert_kml_to_tcx(kml_path, output_path):
    # Parse KML file
    try:
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(kml_path, parser=parser)
        kml_root = tree.getroot()
        
        # Register namespaces
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        for prefix, uri in kml_root.items():
            if 'www.opengis.net/kml' in uri:
                namespaces['kml'] = uri
                break
        
        # Register the namespace with ElementTree
        for prefix, uri in namespaces.items():
            ET.register_namespace(prefix, uri)

        # Create TCX structure
        tcx_root = create_tcx_header()
        activities = ET.SubElement(tcx_root, "Activities")
        activity = ET.SubElement(activities, "Activity")
        activity.set("Sport", "Other")  # You can change this to specific sport type if needed

        # Set the activity ID (current time)
        id_elem = ET.SubElement(activity, "Id")
        id_elem.text = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Create a lap element
        lap = ET.SubElement(activity, "Lap")
        lap.set("StartTime", id_elem.text)

        # Find all Placemarks with timestamps and coordinates
        placemarks = kml_root.findall('.//kml:Placemark', namespaces=namespaces)
        coords_with_time = []
        
        for placemark in placemarks:
            timestamp_elem = placemark.find('.//kml:TimeStamp/kml:when', namespaces=namespaces)
            coord_elem = placemark.find('.//kml:Point/kml:coordinates', namespaces=namespaces)
            
            if timestamp_elem is not None and coord_elem is not None:
                time_str = timestamp_elem.text
                coord_text = coord_elem.text.strip()
                parsed_coord = parse_coordinates(coord_text)
                if parsed_coord and time_str:
                    coords_with_time.append((time_str, parsed_coord))
        
        # Sort by timestamp
        coords_with_time.sort(key=lambda x: x[0])
        coords = [coord for _, coord in coords_with_time]
        start_time = datetime.strptime(coords_with_time[0][0], "%Y-%m-%dT%H:%M:%SZ") if coords_with_time else datetime.utcnow()

        # Create track
        track = ET.SubElement(lap, "Track")
        
        # Add trackpoints
        print(f"Converting {len(coords_with_time)} coordinates...")
        for i, (time_str, (lon, lat, alt)) in enumerate(coords_with_time):
            trackpoint = ET.SubElement(track, "Trackpoint")
            
            # Time - use actual timestamp from KML
            time = ET.SubElement(trackpoint, "Time")
            time.text = time_str
            
            # Position
            position = ET.SubElement(trackpoint, "Position")
            lat_elem = ET.SubElement(position, "LatitudeDegrees")
            lat_elem.text = str(lat)
            lon_elem = ET.SubElement(position, "LongitudeDegrees")
            lon_elem.text = str(lon)
            
            # Altitude
            if alt:
                alt_elem = ET.SubElement(trackpoint, "AltitudeMeters")
                alt_elem.text = str(alt)
            
            # Add speed for this trackpoint if it's not the last point
            if i < len(coords_with_time) - 1:
                next_time_str, next_coords = coords_with_time[i + 1]
                next_lon, next_lat, _ = next_coords
                
                # Calculate time difference in seconds
                current_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
                next_time = datetime.strptime(next_time_str, "%Y-%m-%dT%H:%M:%SZ")
                time_diff = (next_time - current_time).total_seconds()
                
                # Calculate distance to next point
                distance = haversine_distance(lat, lon, next_lat, next_lon)
                
                # Calculate speed in m/s if time difference is not zero
                if time_diff > 0:
                    speed = distance / time_diff
                    
                    # Add speed extension
                    extensions = ET.SubElement(trackpoint, "Extensions")
                    tpx = ET.SubElement(extensions, "{http://www.garmin.com/xmlschemas/ActivityExtension/v2}TPX")
                    speed_elem = ET.SubElement(tpx, "{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Speed")
                    speed_elem.text = f"{speed:.1f}"

        # Calculate total time from timestamps
        start_time = datetime.strptime(coords_with_time[0][0], "%Y-%m-%dT%H:%M:%SZ")
        end_time = datetime.strptime(coords_with_time[-1][0], "%Y-%m-%dT%H:%M:%SZ")
        total_time_seconds = (end_time - start_time).total_seconds()
        
        # Calculate total distance
        distance_meters = 0
        for i in range(len(coords_with_time)-1):
            _, (lon1, lat1, _) = coords_with_time[i]
            _, (lon2, lat2, _) = coords_with_time[i+1]
            distance_meters += haversine_distance(lat1, lon1, lat2, lon2)
        
        # Update lap StartTime with actual start time
        lap.set("StartTime", coords_with_time[0][0])
        
        # Add lap metrics
        lap_time = ET.SubElement(lap, "TotalTimeSeconds")
        lap_time.text = str(total_time_seconds)
        
        lap_distance = ET.SubElement(lap, "DistanceMeters")
        lap_distance.text = str(distance_meters)
        
        # Calculate average speed in m/s
        if total_time_seconds > 0:
            avg_speed = distance_meters / total_time_seconds
            max_speed = avg_speed * 1.5  # Estimate max speed as 150% of average
            
            avg_speed_elem = ET.SubElement(lap, "MaximumSpeed")
            avg_speed_elem.text = f"{max_speed:.1f}"
        
        # Write TCX file
        tree = ET.ElementTree(tcx_root)
        ET.indent(tree, space="  ")  # Pretty print the XML
        tree.write(output_path, encoding="UTF-8", xml_declaration=True)
        
        return True
    except Exception as e:
        print(f"Error converting file: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python kml_to_tcx.py <kml_file>")
        sys.exit(1)

    kml_file = sys.argv[1]
    if not os.path.exists(kml_file):
        print(f"Error: File {kml_file} not found")
        sys.exit(1)

    output_file = os.path.splitext(kml_file)[0] + ".tcx"
    if convert_kml_to_tcx(kml_file, output_file):
        print(f"Successfully converted {kml_file} to {output_file}")
    else:
        print(f"Failed to convert {kml_file}")