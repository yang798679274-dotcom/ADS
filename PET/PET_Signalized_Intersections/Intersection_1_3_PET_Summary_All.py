import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import glob
import os

BASE_DIR = "/Users/xinweiyang/Desktop/TTC_224/PET_Unprotect_left_turn/"

# ====== Load conflict zone polygons ======
def load_polygon(csv_name):
    df = pd.read_csv(os.path.join(BASE_DIR, csv_name))
    return Polygon(list(zip(df["Longitude"], df["Latitude"])))

conflict_zones = {
    "Intersection 1": {"Conflict Zone 1": load_polygon("intersection1_conflict_zone_1.csv")},
    "Intersection 3": {"Conflict Zone 1": load_polygon("intersection3_conflict_zone_1.csv")},
}

# ====== Participant directories ======
intersection_dirs = {
    "Intersection 1": [
        "Inter_1_224_PET_data",
        "Inter_1_230_PET_data",
        "Inter_1_251_PET_data",
        "Inter_1_253_PET_data",
        "Inter_1_263_PET_data",
        "Inter_1_265_PET_data",
        "Inter_1_267_PET_data",
    ],
    "Intersection 3": [
        "Inter_3_224_PET_data",
        "Inter_3_230_PET_data",
        "Inter_3_251_PET_data",
        "Inter_3_253_PET_data",
        "Inter_3_263_PET_data",
        "Inter_3_265_PET_data",
        "Inter_3_267_PET_data",
    ],
}

results = []

for intersection_name, participant_dirs in intersection_dirs.items():
    polygons = conflict_zones[intersection_name]

    for participant_dir in participant_dirs:
        participant_id = participant_dir.split("_")[2]
        dir_path = os.path.join(BASE_DIR, participant_dir)
        csv_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))

        for csv_path in csv_files:
            round_name = os.path.splitext(os.path.basename(csv_path))[0]

            veh_df = pd.read_csv(csv_path)
            veh_df["Time"] = pd.to_datetime(veh_df["Time"])

            ego_ids = veh_df.loc[veh_df["EntityType"] == "ego", "VehicleId"].unique()
            if len(ego_ids) == 0:
                continue
            ego_id = ego_ids[0]

            for poly_name, polygon in polygons.items():

                # Ego entry time into the conflict zone
                ego_df = veh_df[(veh_df["VehicleId"] == ego_id) & (veh_df["EntityType"] == "ego")].copy()
                ego_gdf = gpd.GeoDataFrame(
                    ego_df,
                    geometry=gpd.points_from_xy(ego_df.FrontLong, ego_df.FrontLat),
                    crs="EPSG:4326"
                )
                ego_gdf["in_poly"] = ego_gdf.geometry.apply(lambda p: polygon.covers(p))
                ego_in = ego_gdf[ego_gdf["in_poly"]]

                if ego_in.empty:
                    continue
                ego_first_time = ego_in["Time"].min()

                # Object vehicle processing
                obj_df = veh_df[veh_df["EntityType"] == "object"].copy()
                obj_gdf = gpd.GeoDataFrame(
                    obj_df,
                    geometry=gpd.points_from_xy(obj_df.Longitude, obj_df.Latitude),
                    crs="EPSG:4326"
                )
                obj_gdf["in_poly"] = obj_gdf.geometry.apply(lambda p: polygon.covers(p))

                # PET: object leaves conflict zone → ego enters conflict zone
                min_abs_pet = None
                min_pet = None
                best_obj_exit_time = None
                for vid, group in obj_gdf.groupby("VehicleId"):
                    obj_in = group[group["in_poly"]]
                    if obj_in.empty:
                        continue
                    obj_max_time = obj_in["Time"].max()
                    pet_value = (ego_first_time - obj_max_time).total_seconds()
                    if min_abs_pet is None or abs(pet_value) < min_abs_pet:
                        min_abs_pet = abs(pet_value)
                        min_pet = pet_value
                        best_obj_exit_time = obj_max_time

                if min_pet is not None:
                    results.append({
                        "Intersection": intersection_name,
                        "Participant": participant_id,
                        "Round": round_name,
                        "Polygon": poly_name,
                        "EgoEnterTime": ego_first_time,
                        "ObjExitTime": best_obj_exit_time,
                        "PET(s)": min_pet,
                        "AbsPET(s)": min_abs_pet,
                    })

# ====== Aggregate and export results ======
if not results:
    print("No valid PET records found. Please check that the conflict zone polygons match the trajectory data.")
    exit()

results_df = pd.DataFrame(results).sort_values(
    ["Intersection", "Participant", "Round", "Polygon"]
).reset_index(drop=True)

output_path = os.path.join(BASE_DIR, "Intersection_1_3_PET_Summary_All.csv")
results_df.to_csv(output_path, index=False)

print(f"Processed {len(results_df)} records. Saved to:\n{output_path}")
print()
print(results_df.to_string(index=False))

# ====== Group statistics ======
print("\n====== PET Descriptive Statistics by Intersection ======")
stats = results_df.groupby("Intersection")["PET(s)"].describe().round(3)
print(stats)

print("\n====== PET Descriptive Statistics by Participant ======")
stats_by_participant = results_df.groupby(["Intersection", "Participant"])["PET(s)"].describe().round(3)
print(stats_by_participant)
