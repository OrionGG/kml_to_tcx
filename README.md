# KML to TCX Converter

A Python script that converts KML (Keyhole Markup Language) files to TCX (Training Center XML) format, which is compatible with Garmin Connect and other fitness platforms.

## Features

- Converts KML coordinates and timestamps to TCX format
- Calculates accurate speeds between trackpoints
- Preserves original timestamps from the KML file
- Calculates total distance using the Haversine formula
- Generates proper speed data for Garmin Connect
- Includes altitude data when available

## Usage

```bash
python kml_to_tcx.py <kml_file>
```

The script will generate a TCX file with the same name as the input KML file.

## Requirements

- Python 3.x
- xml.etree.ElementTree (included in Python standard library)
- datetime (included in Python standard library)
- math (included in Python standard library)

## Example

```bash
python kml_to_tcx.py track.kml
```

This will create `track.tcx` in the same directory.

## Output Format

The generated TCX file includes:
- Track points with latitude, longitude, and altitude
- Accurate timestamps from the KML file
- Speed calculations for each track point
- Total distance and time
- Maximum speed estimation

## License

This project is open source and available under the MIT License.