import pandas as pd

# ── 参数 ─────────────────────────────────────────────────────────────
STOP_SPD  = 0.5   # m/s  低于此视为停车
START_SPD = 1.0   # m/s  高于此视为重新启动
LAG_MAX        = 3.0   # 秒   ego 与前车启动时间差 ≤ 此值 → 判为跟车行为
REL_DIST_THRES = 10.0  # 米   ego 启动时前车距离 ≤ 此值且 Δt<0 → 视为近距同步启动（跟车）
VISIT_GAP = 5.0   # 秒   同一路口两段时间断开超过此值视为新的访问

# ── 加载数据 ──────────────────────────────────────────────────────────
df = pd.read_csv('/Users/xinweiyang/Desktop/Signal_awareness/filtered_intersection_data_224_10.csv')

def clean_speed(series):
    return (series.astype(str)
            .str.replace('="', '', regex=False)
            .str.replace('"',  '', regex=False)
            .apply(pd.to_numeric, errors='coerce'))

df['Speed_val'] = clean_speed(df['Speed'])
df['Time_dt']   = pd.to_datetime(df['Time'])

ego = df[df['EntityType'] == 'ego'].copy().sort_values('Time_dt').reset_index(drop=True)
obj = df[df['EntityType'] == 'object'].copy()

# ── 划分路口访问（visit） ─────────────────────────────────────────────
time_gap  = ego['Time_dt'].diff().dt.total_seconds().fillna(0)
group_chg = ego['Intersection_Group'] != ego['Intersection_Group'].shift()
ego['visit_id'] = (group_chg | (time_gap > VISIT_GAP)).cumsum()

total_visits = ego['visit_id'].nunique()
print(f"共识别 {total_visits} 个路口访问\n")

# ── 逐次 visit 检测跟车行为 ────────────────────────────────────────────
bad_visits = set()

for vid, vdf in ego.groupby('visit_id'):
    vdf  = vdf.sort_values('Time_dt').reset_index(drop=True)
    spd  = vdf['Speed_val'].values
    t    = vdf['Time_dt'].values
    intersection = vdf['Intersection_Group'].iloc[0]

    # 该 visit 无停车 → 直接保留
    if not (spd < STOP_SPD).any():
        print(f"Visit {vid} ({intersection}): 无停车 → 保留")
        continue

    is_stopped  = spd < STOP_SPD
    follow_found = False
    i = 0

    while i < len(is_stopped):
        if not is_stopped[i]:
            i += 1
            continue

        # 找停车段起止索引
        stop_start = i
        while i < len(is_stopped) and is_stopped[i]:
            i += 1
        stop_end = i - 1

        stop_rows  = vdf.iloc[stop_start:stop_end + 1]
        front_ids  = stop_rows['FrontCarID'].dropna().unique()

        if len(front_ids) == 0:
            print(f"Visit {vid} ({intersection}): 停车但无前车 → 纯信号停车，保留")
            continue

        front_id      = front_ids[0]
        t_stop_begin  = t[stop_start]
        t_stop_finish = t[stop_end]

        # 在 object 数据中找前车（扩展观察窗口至停车结束后 15s）
        obj_f = obj[obj['VehicleId'] == front_id].copy()
        obj_f['Speed_val'] = clean_speed(obj_f['Speed'])
        obj_f['Time_dt']   = pd.to_datetime(obj_f['Time'])
        obj_f = obj_f.sort_values('Time_dt')

        window = obj_f[
            (obj_f['Time_dt'] >= t_stop_begin) &
            (obj_f['Time_dt'] <= t_stop_finish + pd.Timedelta(seconds=15))
        ]

        if window.empty:
            print(f"Visit {vid} ({intersection}): 找不到前车数据 → 保留")
            continue

        # 前车是否也曾停下
        front_stopped_rows = window[window['Speed_val'] < STOP_SPD]
        if front_stopped_rows.empty:
            print(f"Visit {vid} ({intersection}): 前车未停 → 跟车不适用，保留")
            continue

        t_front_last_stop = front_stopped_rows['Time_dt'].max()

        # 前车停后何时重新启动
        front_resume = window[
            (window['Time_dt'] > t_front_last_stop) &
            (window['Speed_val'] > START_SPD)
        ]
        if front_resume.empty:
            print(f"Visit {vid} ({intersection}): 前车未见重启 → 保留")
            continue

        t_front_start = front_resume['Time_dt'].iloc[0]

        # ego 停车段结束后何时重新启动
        ego_resume = vdf[
            (vdf.index > stop_end) &
            (vdf['Speed_val'] > START_SPD)
        ]
        if ego_resume.empty:
            print(f"Visit {vid} ({intersection}): Ego 未见重启 → 保留")
            continue

        t_ego_start = ego_resume['Time_dt'].iloc[0]
        delta_t     = (t_ego_start - t_front_start).total_seconds()

        # 判断1：经典跟车——前车先启动，ego 在 LAG_MAX 秒内跟进
        is_follow_lag = 0 <= delta_t <= LAG_MAX

        # 判断2：近距同步启动——ego 略早于前车（可能传感器时差），但前车仍在极近距离
        # 取 ego 启动帧的 RelativeDistance
        rel_dist_at_start = ego_resume.iloc[0]['RelativeDistance']
        is_follow_sync = (
            -LAG_MAX <= delta_t < 0
            and pd.notna(rel_dist_at_start)
            and rel_dist_at_start <= REL_DIST_THRES
        )

        is_following = is_follow_lag or is_follow_sync
        verdict = "跟车[剔除整个路口]" if is_following else "信号感知[保留]"

        print(f"Visit {vid} ({intersection}): 前车启动={t_front_start.strftime('%H:%M:%S.%f')[:-3]}, "
              f"Ego启动={t_ego_start.strftime('%H:%M:%S.%f')[:-3]}, "
              f"Δt={delta_t:.1f}s, RelDist@start={rel_dist_at_start:.1f}m → {verdict}")

        if is_following:
            bad_visits.add(vid)
            follow_found = True
            break   # 该 visit 已判为跟车，无需检查更多停车片段

    if not follow_found and vid not in bad_visits:
        pass   # 已在循环内打印

# ── 过滤并保存 ────────────────────────────────────────────────────────
good_ego_times = set(ego[~ego['visit_id'].isin(bad_visits)]['Time'])
result_df = df[df['Time'].isin(good_ego_times)].drop(columns=['Speed_val', 'Time_dt'])

kept_visits = total_visits - len(bad_visits)
print(f"\n{'─'*55}")
print(f"原始路口访问数：{total_visits}")
print(f"跟车访问（已整体剔除）：{len(bad_visits)} 个 → Visit {sorted(bad_visits)}")
print(f"保留纯信号感知访问：{kept_visits} 个")
print(f"原始行数：{len(df)}  →  保留行数：{len(result_df)}")
print(f"{'─'*55}")

result_df.to_csv(
    '/Users/xinweiyang/Desktop/Signal_awareness/signal_only_intersection_data_224_10.csv',
    index=False
)
print("已保存：signal_only_intersection_data_224_10.csv")
