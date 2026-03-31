import pandas as pd
import os
import glob

input_dir = "267_signal_only_intersection_data"
output_dir = "267_signal_only_intersection_data_no_object"
os.makedirs(output_dir, exist_ok=True)


def parse_excel_number(s):
    """Parse Excel text formula like =\"12.34\" to float."""
    if isinstance(s, str):
        return float(s.strip().lstrip('=').strip('"'))
    return float(s)


csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
print(f"Found {len(csv_files)} files\n")

for input_path in csv_files:
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    df = pd.read_csv(input_path)
    df_filtered = df[df['EntityType'] != 'object'].copy()

    # Ensure Speed is numeric (handles Excel ="..." format)
    if df_filtered['Speed'].dtype == object:
        df_filtered['Speed'] = df_filtered['Speed'].apply(parse_excel_number)

    # Initialize accel and jerk columns
    df_filtered['accel'] = 0.0
    df_filtered['jerk'] = 0.0

    intersections = sorted(
        [g for g in df_filtered['Intersection_Group'].unique()
         if isinstance(g, str) and g.startswith('Intersection_')],
        key=lambda x: int(x.split('_')[1])
    )

    for inter in intersections:
        inter_mask = df_filtered['Intersection_Group'] == inter
        ego_mask = inter_mask & (df_filtered['EntityType'] == 'ego')
        seg_idx = df_filtered.index[ego_mask]

        if len(seg_idx) < 2:
            continue

        seg = df_filtered.loc[seg_idx].copy()
        seg = seg.sort_values('Time')

        # dt in seconds from ISO timestamp differences
        dt = pd.to_datetime(seg['Time']).diff().dt.total_seconds()

        seg['accel'] = seg['Speed'].diff() / dt
        seg['jerk'] = seg['accel'].diff() / dt
        seg['accel'] = seg['accel'].fillna(0)
        seg['jerk'] = seg['jerk'].fillna(0)

        df_filtered.loc[seg.index, 'accel'] = seg['accel'].values
        df_filtered.loc[seg.index, 'jerk'] = seg['jerk'].values

    df_filtered.to_csv(output_path, index=False)

    print(f"{filename}: {len(df)} -> {len(df_filtered)} rows (removed {len(df) - len(df_filtered)})")
    for inter in intersections:
        times = df_filtered.loc[df_filtered['Intersection_Group'] == inter, 'Time']
        ego_rows = df_filtered.loc[
            (df_filtered['Intersection_Group'] == inter) & (df_filtered['EntityType'] == 'ego')
        ]
        accel_range = f"accel=[{ego_rows['accel'].min():.3f}, {ego_rows['accel'].max():.3f}]" \
            if len(ego_rows) > 0 else "no ego"
        print(f"  {inter}: {times.iloc[0]}  ->  {times.iloc[-1]}  |  {accel_range}")
    print()

print(f"Done. Output saved to: {output_dir}/")
