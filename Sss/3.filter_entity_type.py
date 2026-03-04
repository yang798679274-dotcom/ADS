import pandas as pd
import os
import glob

input_dir = "224_signal_only_intersection_data"
output_dir = "224_signal_only_intersection_data_no_object"
os.makedirs(output_dir, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
print(f"Found {len(csv_files)} files\n")

for input_path in csv_files:
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    df = pd.read_csv(input_path)
    df_filtered = df[df['EntityType'] != 'object']
    df_filtered.to_csv(output_path, index=False)

    print(f"{filename}: {len(df)} -> {len(df_filtered)} rows (removed {len(df) - len(df_filtered)})")

    intersections = sorted(
        [g for g in df_filtered['Intersection_Group'].unique() if g.startswith('Intersection_')],
        key=lambda x: int(x.split('_')[1])
    )
    for inter in intersections:
        times = df_filtered.loc[df_filtered['Intersection_Group'] == inter, 'Time']
        print(f"  {inter}: {times.iloc[0]}  ->  {times.iloc[-1]}")
    print()

print(f"Done. Output saved to: {output_dir}/")
