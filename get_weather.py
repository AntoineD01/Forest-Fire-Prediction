import pandas as pd
import csv
import os
import time
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from meteostat import Point, Hourly

# --- Configuration ---
INPUT_FILE = "Incendies.csv"
OUTPUT_FILE = "incendies_meteo_safe.csv"
geolocator = Nominatim(user_agent="weather_matcher")

# --- Meteostat requires UTC timestamps ---
def get_weather_data(lat, lon, timestamp):
    try:
        location = Point(lat, lon)
        start = timestamp.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=6)

        data = Hourly(location, start, end)
        data = data.fetch()

        if not data.empty:
            first = data.iloc[0]
            return {
                "temp": first.get("temp", None),
                "humidity": first.get("rhum", None),
                "pressure": first.get("pres", None),
                "wind_speed": first.get("wspd", None),
                "wind_deg": first.get("wdir", None),
                "clouds": first.get("cldc", None),
                "rain_1h": first.get("prcp", None),
                "weather": None  # Meteostat doesn't give weather description
            }
    except Exception as e:
        print(f"[WeatherError] {e}")

    return {
        "temp": None, "humidity": None, "pressure": None,
        "wind_speed": None, "wind_deg": None, "clouds": None,
        "rain_1h": None, "weather": None
    }

# --- Geocoding INSEE code to lat/lon ---
def get_lat_lon(code_insee):
    try:
        location = geolocator.geocode(f"France {code_insee}", timeout=10)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"[GeoError] INSEE {code_insee}: {e}")
    return None, None

# --- Resume: Read already processed rows ---
processed_keys = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            key = row['Code INSEE'] + row['Date de première alerte']
            processed_keys.add(key)

# --- Main processing ---
with open(INPUT_FILE, encoding='utf-8') as infile, open(OUTPUT_FILE, mode='a', encoding='utf-8', newline='', buffering=1) as outfile:
    reader = csv.DictReader(infile, delimiter=';')
    
    # Prepare writer
    fieldnames = reader.fieldnames + ["latitude", "longitude", "temp", "humidity", "pressure",
                                      "wind_speed", "wind_deg", "clouds", "rain_1h", "weather"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')

    # Write header if file is empty
    if os.stat(OUTPUT_FILE).st_size == 0:
        writer.writeheader()
        outfile.flush()

    for i, row in enumerate(reader, start=1):
        try:
            key = row['Code INSEE'] + row['Date de première alerte']
            if key in processed_keys:
                print(f"[{i}] Skipping: Already processed INSEE {row['Code INSEE']}")
                continue

            # Date parsing
            alert_date = pd.to_datetime(row["Date de première alerte"], errors='coerce')
            if pd.isnull(alert_date):
                print(f"[Warning] Invalid date on line {i}")
                alert_date = datetime.utcnow()

            # Geolocation
            lat, lon = get_lat_lon(row["Code INSEE"])
            row["latitude"] = lat
            row["longitude"] = lon

            # Weather
            if lat is not None and lon is not None:
                weather = get_weather_data(lat, lon, alert_date)
            else:
                weather = {key: None for key in ["temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds", "rain_1h", "weather"]}

            row.update(weather)

            # Write line
            writer.writerow(row)
            outfile.flush()

            print(f"[{i}] Saved: INSEE {row['Code INSEE']}")
            time.sleep(1)

        except Exception as e:
            print(f"[Error] Line {i}: {e}")
            continue
