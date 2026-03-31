import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import folium
import re

# # --- Lane groups for Intersection 1 ---
# lane_groups = [
#     [6233, 5999],  # 第一组
#     # [6001, 6395],  # 第二组
#     [6233, 6395],  # 第3组
#     # [6001, 6394]
# ]  # Intersection 1

# lane_groups = [
#     #   [7440, 7334],
#       [7432, 7334],
#       [7333, 7432]
# ] # Intersection 3

# lane_groups = [[7451, 7432],
#                [7333,7432],
#                  [7334, 7432],
#                    [7452, 7432]  # Intersection 3
# ]

lane_groups = [[6369, 5737],
                 [6369, 5733]]# UPLT intersection

# --- Load HD map and build Lane_ID -> Shapely Polygon lookup ---
HD_MAP_PATH = '/Users/xinweiyang/Downloads/HD map files/Semantic_Lanes_Yunji.csv'

df_map = pd.read_csv(HD_MAP_PATH)
df_map['Lane_ID'] = df_map['Lane_ID'].astype(int)


def wkt_to_polygon(wkt_str):
    """Parse MULTIPOLYGON Z WKT and return a 2D Shapely Polygon (lon, lat only)."""
    coords = re.findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)', wkt_str)
    return Polygon([(float(lon), float(lat)) for lon, lat, _ in coords])


lane_polygons = {
    row['Lane_ID']: wkt_to_polygon(row['WKT'])
    for _, row in df_map.iterrows()
}

# Collect all unique lane IDs needed
all_lane_ids = sorted({lid for group in lane_groups for lid in group})

# Verify all lanes exist
for lid in all_lane_ids:
    if lid not in lane_polygons:
        raise ValueError(f"Lane_ID {lid} not found in HD map CSV.")

# --- Compute overlaps and merge into one unified Conflict Zone ---
individual_overlaps = [
    lane_polygons[id_a].intersection(lane_polygons[id_b])
    for id_a, id_b in lane_groups
]
merged_geom = unary_union([g for g in individual_overlaps if not g.is_empty])

conflict_zones = [{
    "name": "Conflict Zone",
    "geom": merged_geom,
    # "csv_name": "intersection1_conflict_zone_1.csv",
    "csv_name": "UPLT_conflict_zone_1.csv"
}]

# --- Legend label mapping ---
# lane_labels = {7432: 'Lane 7432', 7334: 'Lane 7334', 7333: 'Lane 7333'}
# lane_labels = {7451: 'Lane 7451', 7432: 'Lane 7432', 7333: 'Lane 7333', 7334: 'Lane 7334'}
# lane_labels = {6233: 'Lane 6233', 5999: 'Lane 5999', 6001: 'Lane 6001', 6395: 'Lane 6395', 6394: 'Lane 6394'}
lane_labels = {6369: 'Lane 6369', 5737: 'Lane 5737', 5733: 'Lane 5733'}
# --- Matplotlib Plot ---
colors = ['blue', 'green', 'orange', 'red', 'purple']
overlap_colors = ['cyan', 'magenta', 'yellow', 'lime']

fig, ax = plt.subplots(figsize=(10, 10))

for idx, lid in enumerate(all_lane_ids):
    poly = lane_polygons[lid]
    ax.plot(*poly.exterior.xy, color=colors[idx % len(colors)],
            label=lane_labels.get(lid, f'Lane {lid}'), alpha=0.6)


def plot_overlap(ax_obj, geom, color, label):
    if geom.is_empty:
        return
    if geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:
            ax_obj.fill(*part.exterior.xy, color=color, alpha=0.8, label=label)
    else:
        ax_obj.fill(*geom.exterior.xy, color=color, alpha=0.8, label=label)


for idx, zone in enumerate(conflict_zones):
    plot_overlap(ax, zone["geom"], overlap_colors[idx % len(overlap_colors)], zone["name"])

# ax.set_title('Intersection 3: Overlapping Regions of Lane Polygons')
ax.set_title('Unprotected Left Turn: Overlapping Regions of Lane Polygons')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys())

ax.grid(True)
plt.savefig('intersection1_polygon_overlaps_hd.png', dpi=300)

# --- Export overlap coordinates to CSV ---
def overlap_to_dataframe(geom, name):
    rows = []
    if geom.is_empty:
        return pd.DataFrame(columns=["Overlap", "Longitude", "Latitude"])
    polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    for i, part in enumerate(polys):
        label = f"{name}_part{i+1}" if len(polys) > 1 else name
        for lon, lat in part.exterior.coords:
            rows.append({"Overlap": label, "Longitude": lon, "Latitude": lat})
    return pd.DataFrame(rows)


for zone in conflict_zones:
    df = overlap_to_dataframe(zone["geom"], zone["name"])
    df.to_csv(zone["csv_name"], index=False)
    print(f"Area: {zone['geom'].area:.2e}  -> {zone['csv_name']}")

plt.show()

# --- Folium Interactive Map ---
first_poly = lane_polygons[all_lane_ids[0]]
map_center = (first_poly.exterior.coords[0][1], first_poly.exterior.coords[0][0])
m = folium.Map(location=map_center, zoom_start=20)

for idx, lid in enumerate(all_lane_ids):
    poly = lane_polygons[lid]
    folium.Polygon(
        locations=[(lat, lon) for lon, lat in poly.exterior.coords],
        color=colors[idx % len(colors)],
        weight=2,
        fill=True,
        fill_opacity=0.3,
        popup=lane_labels.get(lid, f'Lane {lid}')
    ).add_to(m)

for idx, zone in enumerate(conflict_zones):
    geom = zone["geom"]
    if geom.is_empty:
        continue
    polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    for i, part in enumerate(polys):
        label = f"{zone['name']}_part{i+1}" if len(polys) > 1 else zone["name"]
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in part.exterior.coords],
            color=overlap_colors[idx % len(overlap_colors)],
            weight=2,
            fill=True,
            fill_opacity=0.6,
            popup=label
        ).add_to(m)

m.save("intersection1_polygons_overlaps_map.html")
print("Generated: intersection1_polygons_overlaps_map.html")
