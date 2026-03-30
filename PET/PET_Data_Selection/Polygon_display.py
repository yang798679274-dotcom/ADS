import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, mapping
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import re

# Helper function to parse the coordinates from the provided string.
# This function is not strictly needed for the user's current input
# as the coordinates are already in the correct list-of-tuples format,
# but it's kept for consistency if raw string inputs are ever used again.
def parse_polygon_coords(coord_string):
    # Use a regular expression to find all occurrences of three decimal numbers (lon, lat, alt).
    matches = re.findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)', coord_string)

    # Create a list of (longitude, latitude) tuples from the matches.
    # We only take the first two elements (lon, lat) as altitude is not needed for 2D intersection.
    coords = [(float(lon), float(lat)) for lon, lat, alt in matches]
    return coords

# --- Define polygons as lists of (lon, lat) only (ignoring altitude for intersection) ---
# These coordinates are directly provided by the user in the request.
poly1_coords = [
    (-121.960404968055, 37.7675933950112),
    (-121.960442047484, 37.7675783452625),
    (-121.960458181274, 37.7676033063725),
    (-121.960474480642, 37.7676285249068),
    (-121.960475998685, 37.7676308726355),
    (-121.960478079029, 37.7676340914628),
    (-121.960482985408, 37.7676416821343),
    (-121.960489017524, 37.7676510150128),
    (-121.960496471003, 37.7676625460539),
    (-121.960506928298, 37.7676787263008),
    (-121.960510640036, 37.7676844687056),
    (-121.960514218412, 37.7676900053415),
    (-121.960525128037, 37.767706883569),
    (-121.960541178532, 37.7677317159496),
    (-121.960562802214, 37.7677651705444),
    (-121.960576366714, 37.7677861595284),
    (-121.960538267531, 37.767801623027),
    (-121.960501795484, 37.7677446499118),
    (-121.960490919234, 37.7677276584143),
    (-121.960482475079, 37.7677144692758),
    (-121.960477797468, 37.7677071627257),
    (-121.960469347805, 37.7676939618209),
    (-121.960467330111, 37.7676908093402),
    (-121.960459409288, 37.7676784368776),
    (-121.960457253805, 37.7676750704771),
    (-121.960449882962, 37.7676635572871),
    (-121.960440057774, 37.7676482099944),
    (-121.96043813563, 37.7676452070687),
    (-121.960436399007, 37.7676424959946),
    (-121.960404968055, 37.7675933950112)
]

poly2_coords = [
    (-121.960442047484, 37.7675783452625),
    (-121.960484135074, 37.7675612622141),
    (-121.960491051004, 37.7675719376974),
    (-121.960514167057, 37.7676076226361),
    (-121.960514801646, 37.7676086016255),
    (-121.960515936326, 37.7676103538254),
    (-121.960539373645, 37.7676465300914),
    (-121.9605452149, 37.7676555476351),
    (-121.960549144666, 37.7676616119194),
    (-121.960549621435, 37.7676623479712),
    (-121.960560831698, 37.7676796488478),
    (-121.960580549355, 37.7677100835336),
    (-121.960585197031, 37.7677172582275),
    (-121.960618699931, 37.7677689740504),
    (-121.960576366714, 37.7677861595284),
    (-121.960562802214, 37.7677651705444),
    (-121.960541178532, 37.7677317159496),
    (-121.960525128037, 37.767706883569),
    (-121.960514218412, 37.7676900053415),
    (-121.960510640036, 37.7676844687056),
    (-121.960506928298, 37.7676787263008),
    (-121.960496471003, 37.7676625460539),
    (-121.960489017524, 37.7676510150128),
    (-121.960482985408, 37.7676416821343),
    (-121.960478079029, 37.7676340914628),
    (-121.960475998685, 37.7676308726355),
    (-121.960474480642, 37.7676285249068),
    (-121.960458181274, 37.7676033063725),
    (-121.960442047484, 37.7675783452625)
]
poly3_coords  = [
    (-121.960562802214, 37.7677651705444),
    (-121.960559254284, 37.7677576207007),
    (-121.960555709121, 37.7677474400209),
    (-121.960553294268, 37.7677370530027),
    (-121.960552216232, 37.767728090997),
    (-121.960552029048, 37.76772653913),
    (-121.96055192258, 37.7677159768952),
    (-121.960552528436, 37.7677099246814),
    (-121.960552977122, 37.7677054483355),
    (-121.96055518363, 37.7676950317832),
    (-121.960558525073, 37.767684808204),
    (-121.960560831698, 37.7676796488478),
    (-121.960562975404, 37.7676748539774),
    (-121.960568502887, 37.7676652463342),
    (-121.960575063375, 37.7676560569879),
    (-121.96058146369, 37.7676485768734),
    (-121.960619318132, 37.767705910056),
    (-121.96061918724, 37.7677060954681),
    (-121.960615418117, 37.7677124009817),
    (-121.960613946166, 37.7677153577427),
    (-121.960612868618, 37.7677180403859),
    (-121.960611433258, 37.767723087557),
    (-121.960610211281, 37.7677290315896),
    (-121.960609451456, 37.7677349715871),
    (-121.960609404781, 37.7677400093767),
    (-121.960610374867, 37.7677457339037),
    (-121.960613008726, 37.767755967831),
    (-121.960618699931, 37.7677689740504),
    (-121.960576366714, 37.7677861595284),
    (-121.960562802214, 37.7677651705444)
] ## 6164

poly4_coords = [
    (-121.960367302847, 37.7676086820265),
    (-121.960404968055, 37.7675933950112),
    (-121.960410708458, 37.7676019105554),
    (-121.960417362027, 37.767609997031),
    (-121.960424878547, 37.7676175918057),
    (-121.960433199781, 37.7676246375846),
    (-121.960442193212, 37.7676310304515),
    (-121.960452000465, 37.7676368726),
    (-121.960455325702, 37.7676385125379),
    (-121.960458285599, 37.7676399726001),
    (-121.960460068318, 37.7676408508603),
    (-121.960462334756, 37.7676419686241),
    (-121.960467609504, 37.7676440890226),
    (-121.960473190246, 37.767646330189),
    (-121.960478226989, 37.7676479338517),
    (-121.960484483369, 37.7676499250122),
    (-121.960489017524, 37.7676510150128),
    (-121.960496128251, 37.7676527234948),
    (-121.960498205664, 37.7676530698902),
    (-121.960508035474, 37.7676547059208),
    (-121.960520115544, 37.7676558579813),
    (-121.960532276646, 37.7676561689522),
    (-121.960544424568, 37.7676556371011),
    (-121.9605452149, 37.7676555476351),
    (-121.960556468403, 37.767654267935),
    (-121.960568316072, 37.7676520696547),
    (-121.960579876492, 37.7676490603837),
    (-121.960580781835, 37.767648784464),
    (-121.96058146369, 37.7676485768734),
    (-121.960619318132, 37.767705910056),
    (-121.96060801592, 37.7677077490323),
    (-121.960588757668, 37.7677097873127),
    (-121.960580549355, 37.7677100835336),
    (-121.960569350134, 37.7677104858599),
    (-121.960552528436, 37.7677099246814),
    (-121.960549939865, 37.7677098387565),
    (-121.960530673243, 37.7677078518002),
    (-121.960525128037, 37.767706883569),
    (-121.960511698807, 37.7677045389195),
    (-121.960504009626, 37.7677026261507),
    (-121.960493948996, 37.7677001219455),
    (-121.960493160392, 37.7676999257181),
    (-121.960483601243, 37.7676967977381),
    (-121.960475199421, 37.7676940476929),
    (-121.960472189862, 37.7676928098111),
    (-121.960467330111, 37.7676908093402),
    (-121.960457951519, 37.7676869493024),
    (-121.960455828597, 37.7676858796716),
    (-121.960454505013, 37.7676852127582),
    (-121.960441548762, 37.7676786848883),
    (-121.96043967993, 37.7676775508035),
    (-121.960427354682, 37.7676700686269),
    (-121.960426116304, 37.7676693168425),
    (-121.960413523466, 37.7676601874527),
    (-121.960411770088, 37.767658916489),
    (-121.960408399055, 37.7676560062307),
    (-121.96039862027, 37.7676475632119),
    (-121.96038676782, 37.767635343525),
    (-121.960376542772, 37.7676230129161),
    (-121.960373553526, 37.7676183782557),
    (-121.960372465633, 37.7676166904587),
    (-121.960371922229, 37.7676158483675),
    (-121.960367302847, 37.7676086820265)
]


# Create Shapely Polygon objects from the provided coordinates.
polygon1 = Polygon(poly1_coords)
polygon2 = Polygon(poly2_coords)
polygon3 = Polygon(poly3_coords)
polygon4 = Polygon(poly4_coords)

# --- Compute the intersections for the specified pairs ---
overlap1_4 = polygon1.intersection(polygon4)  # Conflict Zone 1
overlap2_4 = polygon2.intersection(polygon4)  # Conflict Zone 2

# --- Plot all original polygons and their intersections ---
fig, ax = plt.subplots(figsize=(10,10))

# Plot the original polygons with different colors
ax.plot(*polygon1.exterior.xy, color='blue', label='Lane 5737', alpha=0.6)
ax.plot(*polygon2.exterior.xy, color='green', label='Lane 5733', alpha=0.6)
ax.plot(*polygon3.exterior.xy, color='orange', label='Lane 6164', alpha=0.6)
ax.plot(*polygon4.exterior.xy, color='red', label='Lane 6369', alpha=0.6)


# Function to plot overlap regions, handling MultiPolygon cases
def plot_overlap(ax_obj, overlap_geom, color, label):
    if not overlap_geom.is_empty:
        if overlap_geom.geom_type == 'MultiPolygon':
            for geom in overlap_geom.geoms:
                ax_obj.fill(*geom.exterior.xy, color=color, alpha=0.8, label=label)
        else:
            ax_obj.fill(*overlap_geom.exterior.xy, color=color, alpha=0.8, label=label)

# Plot each intersection with a distinct color
plot_overlap(ax, overlap1_4, 'cyan', 'Conflict Zone 1')
plot_overlap(ax, overlap2_4, 'magenta', 'Conflict Zone 2')


# Set the title and axis labels for the plot.
ax.set_title('Unprotected Left Turn: Overlapping Regions of Lane Polygons')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Display the legend to identify all polygons and their intersections.
# Create custom handles for overlaps to avoid duplicate labels if MultiPolygon plots multiple times
handles, labels = ax.get_legend_handles_labels()
unique_labels = {}
for h, l in zip(handles, labels):
    unique_labels[l] = h
ax.legend(unique_labels.values(), unique_labels.keys())


# Add a grid to the plot for better readability of coordinates.
ax.grid(True)

# Save the plot to a PNG file with high resolution.
plt.savefig('polygon_all_overlaps_hd.png', dpi=300)

# Optional: Display the plot.
plt.show()

# ====== Helper function: convert a Polygon or MultiPolygon to a DataFrame ======
def overlap_to_dataframe(overlap_geom, name):
    rows = []
    if overlap_geom.is_empty:
        return pd.DataFrame(columns=["Overlap", "Longitude", "Latitude"])
    if overlap_geom.geom_type == "Polygon":
        coords = list(overlap_geom.exterior.coords)
        for lon, lat in coords:
            rows.append({"Overlap": name, "Longitude": lon, "Latitude": lat})
    elif overlap_geom.geom_type == "MultiPolygon":
        for i, geom in enumerate(overlap_geom.geoms):
            coords = list(geom.exterior.coords)
            for lon, lat in coords:
                rows.append({"Overlap": f"{name}_part{i+1}", "Longitude": lon, "Latitude": lat})
    return pd.DataFrame(rows)

# ====== Convert and export to CSV ======
df1 = overlap_to_dataframe(overlap1_4, "Conflict Zone 1")
df2 = overlap_to_dataframe(overlap2_4, "Conflict Zone 2")

df1.to_csv("conflict_zone_1.csv", index=False)
df2.to_csv("conflict_zone_2.csv", index=False)

print("✅ Generated conflict_zone_1.csv, conflict_zone_2.csv")



import folium

# --- Create Folium map, centered on the first point of poly4 ---
map_center = (poly4_coords[0][1], poly4_coords[0][0])  # (lat, lon)
m = folium.Map(location=map_center, zoom_start=19)

# --- Draw original polygons ---
def add_polygon(fmap, polygon_coords, color, name):
    folium.Polygon(
        locations=[(lat, lon) for lon, lat in polygon_coords],
        color=color,
        weight=2,
        fill=True,
        fill_opacity=0.3,
        popup=name
    ).add_to(fmap)

add_polygon(m, poly1_coords, 'blue', 'Lane 5737')
add_polygon(m, poly2_coords, 'green', 'Lane 5733')
add_polygon(m, poly3_coords, 'orange', 'Lane 6164')
add_polygon(m, poly4_coords, 'red', 'Lane 6369')

# --- Draw overlap regions ---
def add_overlap(fmap, overlap_geom, color, name):
    if overlap_geom.is_empty:
        return
    if overlap_geom.geom_type == 'Polygon':
        coords = [(lat, lon) for lon, lat in overlap_geom.exterior.coords]
        folium.Polygon(
            locations=coords,
            color=color,
            weight=2,
            fill=True,
            fill_opacity=0.5,
            popup=name
        ).add_to(fmap)
    elif overlap_geom.geom_type == 'MultiPolygon':
        for i, geom in enumerate(overlap_geom.geoms):
            coords = [(lat, lon) for lon, lat in geom.exterior.coords]
            folium.Polygon(
                locations=coords,
                color=color,
                weight=2,
                fill=True,
                fill_opacity=0.5,
                popup=f"{name}_part{i+1}"
            ).add_to(fmap)

add_overlap(m, overlap1_4, 'cyan', 'Conflict Zone 1')
add_overlap(m, overlap2_4, 'magenta', 'Conflict Zone 2')

# --- Save as HTML ---
m.save("polygons_overlaps_map.html")
print("✅ Folium map generated: polygons_overlaps_map.html")

# Ego or specified GPS point
ego_lat = 37.7676854181806
ego_lon = -121.960518561106

# Add a red marker on the map
folium.Marker(
    location=(ego_lat, ego_lon),
    popup=f"Ego GPS\n({ego_lat:.12f}, {ego_lon:.12f})",
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(m)

# Save the updated map
m.save("polygons_overlaps_map_with_ego.html")
print("✅ Folium map updated with specified GPS point: polygons_overlaps_map_with_ego.html")
