import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import glob
import os

# ====== Load conflict zone polygons ======
poly1_df = pd.read_csv("/Users/xinweiyang/overlap1_4.csv")
poly2_df = pd.read_csv("/Users/xinweiyang/overlap2_4.csv")
poly1 = Polygon(list(zip(poly1_df["Longitude"], poly1_df["Latitude"])))
poly2 = Polygon(list(zip(poly2_df["Longitude"], poly2_df["Latitude"])))
polygons = {
    "overlap1_4": poly1,
    "overlap2_4": poly2
}

# ====== Participant directory list ======
BASE_DIR = "/Users/xinweiyang/Desktop/TTC_224/PET_Unprotect_left_turn"
participant_dirs = [
    "224_PET_data",
    "230_PET_data",
    "251_PET_data",
    "253_PET_data",
    "263_PET_data",
    "265_PET_data",
    "267_PET_data",
]

results = []

for participant_dir in participant_dirs:
    participant_id = participant_dir.split("_")[0]
    dir_path = os.path.join(BASE_DIR, participant_dir)
    csv_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))

    for csv_path in csv_files:
        round_name = os.path.splitext(os.path.basename(csv_path))[0]  # e.g. "224_round1_unprotected"

        veh_df = pd.read_csv(csv_path)
        veh_df["Time"] = pd.to_datetime(veh_df["Time"])

        ego_ids = veh_df.loc[veh_df["EntityType"] == "ego", "VehicleId"].unique()
        if len(ego_ids) == 0:
            continue
        ego_id = ego_ids[0]

        for poly_name, polygon in polygons.items():

            # ====== Ego entry time ======
            ego_df = veh_df[(veh_df["VehicleId"] == ego_id) & (veh_df["EntityType"] == "ego")].copy()
            ego_gdf = gpd.GeoDataFrame(
                ego_df,
                geometry=gpd.points_from_xy(ego_df.FrontLong, ego_df.FrontLat),
                crs="EPSG:4326"
            )
            ego_gdf["in_poly"] = ego_gdf.geometry.apply(lambda p: polygon.covers(p))
            ego_in = ego_gdf[ego_gdf["in_poly"]]

            if ego_in.empty:
                # Ego did not enter this polygon, skip
                continue
            ego_first_time = ego_in["Time"].min()

            # ====== Object vehicle processing ======
            obj_df = veh_df[veh_df["EntityType"] == "object"].copy()
            obj_gdf = gpd.GeoDataFrame(
                obj_df,
                geometry=gpd.points_from_xy(obj_df.Longitude, obj_df.Latitude),
                crs="EPSG:4326"
            )
            obj_gdf["in_poly"] = obj_gdf.geometry.apply(lambda p: polygon.covers(p))

            # PET definition: time interval from object leaving the conflict zone (obj_max_time)
            # to ego entering the conflict zone (ego_min_time).
            # Select the object with the smallest absolute time difference to ego as the PET for this interval.
            min_abs_pet = None
            min_pet = None
            best_obj_exit_time = None
            for vid, group in obj_gdf.groupby("VehicleId"):
                obj_in = group[group["in_poly"]]
                if obj_in.empty:
                    continue
                obj_max_time = obj_in["Time"].max()  # Last timestamp of object inside the conflict zone
                pet_value = (ego_first_time - obj_max_time).total_seconds()
                if min_abs_pet is None or abs(pet_value) < min_abs_pet:
                    min_abs_pet = abs(pet_value)
                    min_pet = pet_value
                    best_obj_exit_time = obj_max_time

            if min_pet is not None:
                results.append({
                    "Participant": participant_id,
                    "Round": round_name,
                    "Polygon": poly_name,
                    "EgoEnterTime": ego_first_time,
                    "ObjExitTime": best_obj_exit_time,
                    "PET(s)": min_pet,
                    "AbsPET(s)": min_abs_pet,
                })

# ====== Aggregate and export results ======
results_df = pd.DataFrame(results).sort_values(
    ["Participant", "Round", "Polygon"]
).reset_index(drop=True)

output_path = os.path.join(BASE_DIR, "PET_Summary_ALL_Polygon.csv")
results_df.to_csv(output_path, index=False)

print(f"Processed {len(results_df)} records. Saved to:\n{output_path}")
print()
print(results_df.to_string(index=False))
