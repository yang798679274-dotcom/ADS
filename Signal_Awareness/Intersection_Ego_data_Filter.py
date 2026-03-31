import pandas as pd

# 1. Load data
df = pd.read_csv('/Users/xinweiyang/Desktop/TTC_224_0811/gaosu_matched_pairs_final_front/10_matched_front_original.csv')

# 2. Data cleaning: Lane_ID may appear as 5937.0 (float) or ="5563" format; normalize to plain integer strings
df['Lane_ID'] = (
    df['Lane_ID'].astype(str)
    .str.replace('="', '', regex=False)
    .str.replace('"', '', regex=False)
    .str.replace(r'\.0$', '', regex=True)   # strip trailing .0 from float values
)

# 3. Define intersection configuration (Lane_ID lists)
intersections = {
    "Intersection_1": ["6470", "6233"],
    "Intersection_2": ["5674", "5666"],
    "Intersection_3": ["7368", "7432"],
    "Intersection_4": ["5587", "6119"],
    "Intersection_5": ["5563", "5564"],
    "Intersection_6": ["7474", "7443", "7329", "7444"],
    "Intersection_7": ["7609", "5668"]
}

# Collect all target Lane_IDs
all_target_lanes = [lane for lanes in intersections.values() for lane in lanes]

# 4. Filtering logic
# Step 1: Find timestamps where the ego vehicle is on a target lane
# EntityType == 'ego' is the correct ego identifier (VehicleId is numeric, does not contain 'ego')
# Lane_ID may be a multi-value string like "5563;5564", so use str.contains to match any target lane
lane_pattern = '|'.join(all_target_lanes)
ego_mask = (
    (df['EntityType'] == 'ego') &
    (df['Lane_ID'].str.contains(lane_pattern, na=False))
)
target_times = df.loc[ego_mask, 'Time'].unique()

# Step 2: Extract data for all vehicles (ego + surrounding objects) at those timestamps
final_df = df[df['Time'].isin(target_times)].copy()

# 5. Label which intersection each row belongs to (Lane_ID may be multi-value; use first match)
def mark_intersection(lane_str):
    for part in str(lane_str).split(';'):
        part = part.strip()
        for name, lanes in intersections.items():
            if part in lanes:
                return name
    return "Other"

final_df['Intersection_Group'] = final_df['Lane_ID'].apply(mark_intersection)

# Save result
final_df.to_csv('filtered_intersection_data_224_10.csv', index=False)
print(f"Filtering complete. Extracted {len(final_df)} records across {len(target_times)} time frames.")
