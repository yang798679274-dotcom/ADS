import pandas as pd
import os
import glob

def clean_excel_format(val):
    """
    Helper function: clean Excel format artifacts.
    Converts '="123.45"' to '123.45'.
    """
    if isinstance(val, str) and val.startswith('="'):
        return val.replace('="', '').replace('"', '')
    return val

def process_folder(folder_path):
    # Define target Lane_IDs
    START_LANE_ID = 6470
    END_LANE_ID = 6233 # Intersection_1

    # START_LANE_ID = 7368
    # END_LANE_ID   = 7432 # Intersection_3

    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in path: {folder_path}")
        return

    print(f"Found {len(csv_files)} CSV file(s), starting processing...")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)

        # Skip already-processed files (avoid re-processing generated files)
        if file_name.startswith("sliced_"):
            continue

        print(f"\n--- Processing: {file_name} ---")

        try:
            df = pd.read_csv(file_path)

            # 1. Data preprocessing
            # Ensure the Time column is in datetime format
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
            else:
                print(f"Error: 'Time' column not found, skipping.")
                continue

            # Ensure Lane_ID is numeric
            if 'Lane_ID' in df.columns:
                df['Lane_ID_Clean'] = pd.to_numeric(df['Lane_ID'], errors='coerce')
            else:
                print(f"Error: 'Lane_ID' column not found, skipping.")
                continue

            # Handle EntityType (avoid 'float object has no attribute lower' error)
            df['EntityType_Str'] = df['EntityType'].astype(str).str.lower()

            # 2. Filter ego vehicle data to determine the time range
            ego_df = df[df['EntityType_Str'] == 'ego']

            if ego_df.empty:
                print(f"Warning: No data with EntityType 'ego' found.")
                continue

            # 3. Find slice boundaries
            # Start: first occurrence of Lane_ID == START_LANE_ID
            start_rows = ego_df[ego_df['Lane_ID_Clean'] == START_LANE_ID]
            if start_rows.empty:
                print(f"Skipping: start Lane_ID {START_LANE_ID} not found in ego data.")
                continue
            start_time = start_rows.iloc[0]['Time']

            # End: last occurrence of Lane_ID == END_LANE_ID
            end_rows = ego_df[ego_df['Lane_ID_Clean'] == END_LANE_ID]
            if end_rows.empty:
                print(f"Skipping: end Lane_ID {END_LANE_ID} not found in ego data.")
                continue
            end_time = end_rows.iloc[-1]['Time']

            # 4. Validate time logic
            if start_time > end_time:
                print(f"Warning: start time is later than end time, skipping this file.")
                continue

            # 5. Slice the raw data (includes all objects)
            sliced_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)].copy()

            # Filter out rows where VehicleType is 'pedestrian'
            if 'VehicleType' in sliced_df.columns:
                sliced_df = sliced_df[sliced_df['VehicleType'].astype(str).str.lower() != 'pedestrian']

            # Drop auxiliary columns
            sliced_df = sliced_df.drop(columns=['Lane_ID_Clean', 'EntityType_Str'])

            # Clean ="..." format from data
            sliced_df = sliced_df.applymap(clean_excel_format)

            # 6. Save file to the same folder
            output_filename = f"sliced_{file_name}"
            output_path = os.path.join(folder_path, output_filename)

            sliced_df.to_csv(output_path, index=False)
            print(f"✅ Successfully sliced, cleaned, and saved to: {output_filename} ({len(sliced_df)} rows)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error occurred while processing file: {e}")

if __name__ == "__main__":
    # Specify the target folder path
    target_folder = '/Users/xinweiyang/Desktop/TTC_253_0209/HB'

    if os.path.exists(target_folder):
        process_folder(target_folder)
    else:
        print(f"Error: Folder path does not exist: {target_folder}")
