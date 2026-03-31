import pandas as pd
import os
import glob

# ── Parameters ───────────────────────────────────────────────────────
STOP_SPD       = 0.5   # m/s  below this threshold → considered stopped
START_SPD      = 0.5   # m/s  above this threshold → considered resumed
LAG_MAX        = 5.0   # s    ego start lag behind front vehicle ≤ this → classified as car-following
REL_DIST_THRES = 15.0  # m    front vehicle distance ≤ this at ego start and Δt<0 → near-synchronous start (car-following)
VISIT_GAP      = 5.0   # s    time gap > this within the same intersection → treated as a new visit

# ── I/O paths ─────────────────────────────────────────────────────────
INPUT_DIR  = "253_filtered_intersection_data"
OUTPUT_DIR = "253_signal_only_no_object"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────
def clean_speed(series):
    """Parse Speed column that may contain Excel ="..." format or plain floats."""
    return (series.astype(str)
            .str.replace('="', '', regex=False)
            .str.replace('"',  '', regex=False)
            .apply(pd.to_numeric, errors='coerce'))


# ── Main loop ─────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
print(f"Found {len(csv_files)} file(s) in '{INPUT_DIR}'\n")

for input_path in csv_files:
    filename = os.path.basename(input_path)
    output_path = os.path.join(OUTPUT_DIR, filename)
    print(f"{'═'*60}")
    print(f"Processing: {filename}")
    print(f"{'═'*60}")

    # ── Step 1: Load ──────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    df['Speed_val'] = clean_speed(df['Speed'])
    df['Time_dt']   = pd.to_datetime(df['Time'])

    ego = df[df['EntityType'] == 'ego'].copy().sort_values('Time_dt').reset_index(drop=True)
    obj = df[df['EntityType'] == 'object'].copy()

    # ── Step 2: Segment intersection visits ───────────────────────────
    time_gap  = ego['Time_dt'].diff().dt.total_seconds().fillna(0)
    group_chg = ego['Intersection_Group'] != ego['Intersection_Group'].shift()
    ego['visit_id'] = (group_chg | (time_gap > VISIT_GAP)).cumsum()

    total_visits = ego['visit_id'].nunique()
    print(f"Total intersection visits identified: {total_visits}\n")

    # ── Step 3: Detect car-following behavior per visit ───────────────
    bad_visits = set()

    for vid, vdf in ego.groupby('visit_id'):
        vdf         = vdf.sort_values('Time_dt').reset_index(drop=True)
        spd         = vdf['Speed_val'].values
        t           = vdf['Time_dt'].values
        intersection = vdf['Intersection_Group'].iloc[0]

        # No stop in this visit → keep directly
        if not (spd < STOP_SPD).any():
            print(f"  Visit {vid} ({intersection}): No stop → keep")
            continue

        is_stopped   = spd < STOP_SPD
        follow_found = False
        i = 0

        while i < len(is_stopped):
            if not is_stopped[i]:
                i += 1
                continue

            # Find start/end indices of the stopped segment
            stop_start = i
            while i < len(is_stopped) and is_stopped[i]:
                i += 1
            stop_end = i - 1

            stop_rows = vdf.iloc[stop_start:stop_end + 1]
            front_ids = stop_rows['FrontCarID'].dropna().unique()

            if len(front_ids) == 0:
                print(f"  Visit {vid} ({intersection}): Stopped but no front vehicle → pure signal stop, keep")
                continue

            front_id      = front_ids[0]
            t_stop_begin  = t[stop_start]
            t_stop_finish = t[stop_end]

            # Look up front vehicle in object data (extend observation window to 15s after stop ends)
            obj_f = obj[obj['VehicleId'] == front_id].copy()
            obj_f['Speed_val'] = clean_speed(obj_f['Speed'])
            obj_f['Time_dt']   = pd.to_datetime(obj_f['Time'])
            obj_f = obj_f.sort_values('Time_dt')

            window = obj_f[
                (obj_f['Time_dt'] >= t_stop_begin) &
                (obj_f['Time_dt'] <= t_stop_finish + pd.Timedelta(seconds=15))
            ]

            if window.empty:
                print(f"  Visit {vid} ({intersection}): Front vehicle data not found → keep")
                continue

            # Check whether the front vehicle also stopped
            front_stopped_rows = window[window['Speed_val'] < STOP_SPD]
            if front_stopped_rows.empty:
                print(f"  Visit {vid} ({intersection}): Front vehicle did not stop → car-following not applicable, keep")
                continue

            t_front_last_stop = front_stopped_rows['Time_dt'].max()

            # Find when the front vehicle resumed after stopping
            front_resume = window[
                (window['Time_dt'] > t_front_last_stop) &
                (window['Speed_val'] > START_SPD)
            ]
            if front_resume.empty:
                print(f"  Visit {vid} ({intersection}): Front vehicle never resumed → keep")
                continue

            t_front_start = front_resume['Time_dt'].iloc[0]

            # Find when ego resumed after its stopped segment
            ego_resume = vdf[
                (vdf.index > stop_end) &
                (vdf['Speed_val'] > START_SPD)
            ]
            if ego_resume.empty:
                print(f"  Visit {vid} ({intersection}): Ego never resumed → keep")
                continue

            t_ego_start = ego_resume['Time_dt'].iloc[0]
            delta_t     = (t_ego_start - t_front_start).total_seconds()

            # Criterion 1: Classic car-following — front vehicle starts first, ego follows within LAG_MAX seconds
            is_follow_lag = 0 <= delta_t <= LAG_MAX

            # Criterion 2: Near-synchronous start — ego starts slightly before front vehicle (possible sensor lag),
            # but front vehicle is still at very close range
            rel_dist_at_start = ego_resume.iloc[0]['RelativeDistance']
            is_follow_sync = (
                -LAG_MAX <= delta_t < 0
                and pd.notna(rel_dist_at_start)
                and rel_dist_at_start <= REL_DIST_THRES
            )

            is_following = is_follow_lag or is_follow_sync
            verdict = "Car-following [remove entire visit]" if is_following else "Signal-aware [keep]"

            print(f"  Visit {vid} ({intersection}): FrontStart={t_front_start.strftime('%H:%M:%S.%f')[:-3]}, "
                  f"EgoStart={t_ego_start.strftime('%H:%M:%S.%f')[:-3]}, "
                  f"Δt={delta_t:.1f}s, RelDist@start={rel_dist_at_start:.1f}m → {verdict}")

            if is_following:
                bad_visits.add(vid)
                follow_found = True
                break   # This visit is classified as car-following; no need to check further stop segments

    # ── Step 4: Remove car-following visits ───────────────────────────
    good_ego_times = set(ego[~ego['visit_id'].isin(bad_visits)]['Time'])
    signal_df = df[df['Time'].isin(good_ego_times)].drop(columns=['Speed_val', 'Time_dt'])

    kept_visits = total_visits - len(bad_visits)
    print(f"\n{'─'*55}")
    print(f"Total intersection visits:              {total_visits}")
    print(f"Car-following visits (removed):         {len(bad_visits)} → Visit {sorted(bad_visits)}")
    print(f"Pure signal-aware visits retained:      {kept_visits}")
    print(f"Rows after signal filter: {len(df)} → {len(signal_df)}")
    print(f"{'─'*55}\n")

    # ── Step 5: Remove object rows, compute accel & jerk for ego ──────
    result_df = signal_df[signal_df['EntityType'] != 'object'].copy()

    # Ensure Speed is numeric
    result_df['Speed'] = clean_speed(result_df['Speed'])

    result_df['accel'] = 0.0
    result_df['jerk']  = 0.0

    intersections = sorted(
        [g for g in result_df['Intersection_Group'].unique()
         if isinstance(g, str) and g.startswith('Intersection_')],
        key=lambda x: int(x.split('_')[1])
    )

    for inter in intersections:
        ego_mask = (result_df['Intersection_Group'] == inter) & (result_df['EntityType'] == 'ego')
        seg_idx  = result_df.index[ego_mask]

        if len(seg_idx) < 2:
            continue

        seg = result_df.loc[seg_idx].copy().sort_values('Time')
        dt  = pd.to_datetime(seg['Time']).diff().dt.total_seconds()

        seg['accel'] = seg['Speed'].diff() / dt
        seg['jerk']  = seg['accel'].diff() / dt
        seg['accel'] = seg['accel'].fillna(0)
        seg['jerk']  = seg['jerk'].fillna(0)

        result_df.loc[seg.index, 'accel'] = seg['accel'].values
        result_df.loc[seg.index, 'jerk']  = seg['jerk'].values

    # ── Step 6: Save ──────────────────────────────────────────────────
    result_df.to_csv(output_path, index=False)

    print(f"Rows after object removal: {len(signal_df)} → {len(result_df)}")
    for inter in intersections:
        times    = result_df.loc[result_df['Intersection_Group'] == inter, 'Time']
        ego_rows = result_df.loc[
            (result_df['Intersection_Group'] == inter) & (result_df['EntityType'] == 'ego')
        ]
        accel_range = (f"accel=[{ego_rows['accel'].min():.3f}, {ego_rows['accel'].max():.3f}]"
                       if len(ego_rows) > 0 else "no ego")
        print(f"  {inter}: {times.iloc[0]}  ->  {times.iloc[-1]}  |  {accel_range}")
    print(f"Saved → {output_path}\n")

print(f"Done. All outputs saved to: {OUTPUT_DIR}/")
