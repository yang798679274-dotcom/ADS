import pandas as pd

# ── Parameters ───────────────────────────────────────────────────────
STOP_SPD  = 0.5   # m/s  below this threshold → considered stopped
START_SPD = 0.5   # m/s  above this threshold → considered resumed
LAG_MAX        = 5.0   # s    ego start lag behind front vehicle ≤ this → classified as car-following
REL_DIST_THRES = 15.0  # m    front vehicle distance ≤ this at ego start and Δt<0 → near-synchronous start (car-following)
VISIT_GAP = 5.0   # s    time gap > this within the same intersection → treated as a new visit

# ── Load data ─────────────────────────────────────────────────────────
df = pd.read_csv('/Users/xinweiyang/Desktop/Signal_awareness/253_filtered_intersection_data/filtered_intersection_data_3.csv')

def clean_speed(series):
    return (series.astype(str)
            .str.replace('="', '', regex=False)
            .str.replace('"',  '', regex=False)
            .apply(pd.to_numeric, errors='coerce'))

df['Speed_val'] = clean_speed(df['Speed'])
df['Time_dt']   = pd.to_datetime(df['Time'])

ego = df[df['EntityType'] == 'ego'].copy().sort_values('Time_dt').reset_index(drop=True)
obj = df[df['EntityType'] == 'object'].copy()

# ── Segment intersection visits ──────────────────────────────────────
time_gap  = ego['Time_dt'].diff().dt.total_seconds().fillna(0)
group_chg = ego['Intersection_Group'] != ego['Intersection_Group'].shift()
ego['visit_id'] = (group_chg | (time_gap > VISIT_GAP)).cumsum()

total_visits = ego['visit_id'].nunique()
print(f"Total intersection visits identified: {total_visits}\n")

# ── Detect car-following behavior per visit ────────────────────────────
bad_visits = set()

for vid, vdf in ego.groupby('visit_id'):
    vdf  = vdf.sort_values('Time_dt').reset_index(drop=True)
    spd  = vdf['Speed_val'].values
    t    = vdf['Time_dt'].values
    intersection = vdf['Intersection_Group'].iloc[0]

    # No stop in this visit → keep directly
    if not (spd < STOP_SPD).any():
        print(f"Visit {vid} ({intersection}): No stop → keep")
        continue

    is_stopped  = spd < STOP_SPD
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

        stop_rows  = vdf.iloc[stop_start:stop_end + 1]
        front_ids  = stop_rows['FrontCarID'].dropna().unique()

        if len(front_ids) == 0:
            print(f"Visit {vid} ({intersection}): Stopped but no front vehicle → pure signal stop, keep")
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
            print(f"Visit {vid} ({intersection}): Front vehicle data not found → keep")
            continue

        # Check whether the front vehicle also stopped
        front_stopped_rows = window[window['Speed_val'] < STOP_SPD]
        if front_stopped_rows.empty:
            print(f"Visit {vid} ({intersection}): Front vehicle did not stop → car-following not applicable, keep")
            continue

        t_front_last_stop = front_stopped_rows['Time_dt'].max()

        # Find when the front vehicle resumed after stopping
        front_resume = window[
            (window['Time_dt'] > t_front_last_stop) &
            (window['Speed_val'] > START_SPD)
        ]
        if front_resume.empty:
            print(f"Visit {vid} ({intersection}): Front vehicle never resumed → keep")
            continue

        t_front_start = front_resume['Time_dt'].iloc[0]

        # Find when ego resumed after its stopped segment
        ego_resume = vdf[
            (vdf.index > stop_end) &
            (vdf['Speed_val'] > START_SPD)
        ]
        if ego_resume.empty:
            print(f"Visit {vid} ({intersection}): Ego never resumed → keep")
            continue

        t_ego_start = ego_resume['Time_dt'].iloc[0]
        delta_t     = (t_ego_start - t_front_start).total_seconds()

        # Criterion 1: Classic car-following — front vehicle starts first, ego follows within LAG_MAX seconds
        is_follow_lag = 0 <= delta_t <= LAG_MAX

        # Criterion 2: Near-synchronous start — ego starts slightly before front vehicle (possible sensor lag),
        # but front vehicle is still at very close range
        # Use RelativeDistance from ego's first resumed frame
        rel_dist_at_start = ego_resume.iloc[0]['RelativeDistance']
        is_follow_sync = (
            -LAG_MAX <= delta_t < 0
            and pd.notna(rel_dist_at_start)
            and rel_dist_at_start <= REL_DIST_THRES
        )

        is_following = is_follow_lag or is_follow_sync
        verdict = "Car-following [remove entire visit]" if is_following else "Signal-aware [keep]"

        print(f"Visit {vid} ({intersection}): FrontStart={t_front_start.strftime('%H:%M:%S.%f')[:-3]}, "
              f"EgoStart={t_ego_start.strftime('%H:%M:%S.%f')[:-3]}, "
              f"Δt={delta_t:.1f}s, RelDist@start={rel_dist_at_start:.1f}m → {verdict}")

        if is_following:
            bad_visits.add(vid)
            follow_found = True
            break   # This visit is classified as car-following; no need to check further stop segments

    if not follow_found and vid not in bad_visits:
        pass   # 已在循环内打印

# ── Filter and save ───────────────────────────────────────────────────
good_ego_times = set(ego[~ego['visit_id'].isin(bad_visits)]['Time'])
result_df = df[df['Time'].isin(good_ego_times)].drop(columns=['Speed_val', 'Time_dt'])

kept_visits = total_visits - len(bad_visits)
print(f"\n{'─'*55}")
print(f"Total intersection visits: {total_visits}")
print(f"Car-following visits (removed entirely): {len(bad_visits)} → Visit {sorted(bad_visits)}")
print(f"Pure signal-aware visits retained: {kept_visits}")
print(f"Original rows: {len(df)}  →  Retained rows: {len(result_df)}")
print(f"{'─'*55}")

result_df.to_csv(
    '/Users/xinweiyang/Desktop/Signal_awareness/signal_only_intersection_data_224_10.csv',
    index=False
)
print("Saved: signal_only_intersection_data_224_10.csv")
