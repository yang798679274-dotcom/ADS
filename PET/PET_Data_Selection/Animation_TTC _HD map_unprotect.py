"""
make_trajectory_animation.py

Purpose: Read trajectory CSV and HD map CSV, generate a time-slider playback HTML (folium).
Output: trajectories_animation.html
"""

import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
import shapely.wkt
from shapely.geometry import mapping
import random
from branca.element import MacroElement
from jinja2 import Template

# ====== Configuration ======
vehicle_csv = "/mnt/data/230_round3_lanechange.csv"   # Trajectory CSV (contains Time, Latitude, Longitude, VehicleId)
hdmap_csv    = "/mnt/data/Semantic_Lanes.csv"         # HD map CSV (contains WKT column)
time_col     = "Time"        # Time column name (ISO8601 format preferred)
lat_col      = "Latitude"
lon_col      = "Longitude"
vid_col      = "VehicleId"        # Change to the actual column name if your file uses a different one
wkt_col      = "WKT"         # WKT column name in the HD map CSV (modify if different)
dir_col      = "Direction"
output_html  = "/Users/xinweiyang/Desktop/TTC_224/PET_Unprotect_left_turn/trajectories_animation_253_R1_Inter_3.html"

map_start_zoom = 15

# ====== Load data ======
df = pd.read_csv('/Users/xinweiyang/Desktop/TTC_224/PET_Unprotect_left_turn/Inter_3_253_PET_data/sliced_1.csv')
df_hd = pd.read_csv('/Users/xinweiyang/Downloads/HD map files/Semantic_Lanes.csv')


df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col, lat_col, lon_col]).copy()
df[time_col] = df[time_col].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").str.replace(r"000Z$", "Z", regex=True)


# Dynamically select GPS columns based on EntityType
def select_lat_lon(row):
    if row.get('EntityType', '').lower() == 'ego':
        return pd.Series([row.get('FrontLat'), row.get('FrontLong')])
    else:
        return pd.Series([row.get('Latitude'), row.get('Longitude')])

df[[lat_col, lon_col]] = df.apply(select_lat_lon, axis=1)
df = df.dropna(subset=[lat_col, lon_col])

# 🚩 Key: ensure VehicleId is a string
df[vid_col] = df[vid_col].astype(str)

# ====== Color assignment: generate a unique color for each VehicleId ======
vehicle_ids = sorted(df[vid_col].unique().tolist())
def gen_color(seed=None):
    random.seed(seed)
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

color_map = {vid: gen_color(i) for i, vid in enumerate(vehicle_ids)}

# ====== Map center ======
center_lat = df[lat_col].mean()
center_lon = df[lon_col].mean()

# ====== Build GeoJSON features ======
features = []
for _, row in df.iterrows():
    vid = str(row[vid_col])
    t = row[time_col]
    lat = float(row[lat_col])
    lon = float(row[lon_col])
    direction = float(row[dir_col])   # 🚩 Added: read Direction column

    feat = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "time": t,
            "popup": f"VehicleId: {vid}<br>Time: {t}<br> Direction: {direction}",
            "icon": "circle",
            "style": {
                "color": color_map.get(vid, "#3388ff"),
                "fillColor": color_map.get(vid, "#3388ff"),
                "radius": 4
            },
            "vehicle_id": vid,
            "direction": direction
        }
    }

    features.append(feat)

feature_collection = {"type": "FeatureCollection", "features": features}

# ====== Create map ======
m = folium.Map(location=[df.iloc[0]["Latitude"], df.iloc[0]["Longitude"]],zoom_start=18, max_zoom=22, tiles="OpenStreetMap")


# HD map polygons
if wkt_col in df_hd.columns:
    for i, wkt in enumerate(df_hd[wkt_col].dropna().unique()):
        try:
            geom = shapely.wkt.loads(wkt)
            geojson = mapping(geom)
            folium.GeoJson(
                geojson,
                name=f"hdmap_{i}",
                style_function=lambda feature: {"color": "#BEBEBE", "weight": 0.6, "fillOpacity": 0.1}
            ).add_to(m)
        except Exception:
            continue

# Trajectory animation
TimestampedGeoJson(
    data=feature_collection,
    transition_time=200,
    period='PT1S',
    add_last_point=True,
    auto_play=False,
    loop=False,
    max_speed=10,
    loop_button=True,
    date_options='YYYY-MM-DD HH:mm:ss',
    time_slider_drag_update=True
).add_to(m)

# Legend
from branca.element import MacroElement
from jinja2 import Template

def make_legend_html(color_map, max_items=12):
    items = list(color_map.items())[:max_items]
    rows = "".join(
        f'<li><span style="display:inline-block;width:12px;height:12px;background:{c};margin-right:6px;"></span>{vid}</li>'
        for vid, c in items
    )
    more = f"<div>... and {len(color_map)-max_items} more vehicles</div>" if len(color_map) > max_items else ""
    return f"""
    <div style='position: fixed; bottom: 50px; left: 10px; z-index:9999; background:#fff; padding:8px; border:1px solid #ccc;'>
      <strong>Vehicle Legend</strong>
      <ul style='list-style:none; padding-left:0; margin:6px 0 0 0; font-size:12px;'>{rows}</ul>
      {more}
    </div>
    """

legend = MacroElement()
legend._template = Template(make_legend_html(color_map, max_items=12))
m.get_root().add_child(legend)

# Save
m.save(output_html)
print(f"Saved animation HTML -> {output_html}")