import pandas as pd

# 1. 加载数据
df = pd.read_csv('/Users/xinweiyang/Desktop/TTC_224_0811/gaosu_matched_pairs_final_front/10_matched_front_original.csv')

# 2. 数据清洗：Lane_ID 可能是 5937.0 (float) 或 ="5563" 格式，统一转为纯整数字符串
df['Lane_ID'] = (
    df['Lane_ID'].astype(str)
    .str.replace('="', '', regex=False)
    .str.replace('"', '', regex=False)
    .str.replace(r'\.0$', '', regex=True)   # 去掉 float 尾部的 .0
)

# 3. 定义交叉口配置 (Lane_ID 列表)
intersections = {
    "Intersection_1": ["6470", "6233"],
    "Intersection_2": ["5674", "5666"],
    "Intersection_3": ["7368", "7432"],
    "Intersection_4": ["5587", "6119"],
    "Intersection_5": ["5563", "5564"],
    "Intersection_6": ["7474", "7443", "7329", "7444"],
    "Intersection_7": ["7609", "5668"]
}

# 提取所有目标 Lane_ID
all_target_lanes = [lane for lanes in intersections.values() for lane in lanes]

# 4. 筛选逻辑
# 第一步：找出 Ego 车辆在目标车道上的时间点
# EntityType == 'ego' 是正确的 ego 标识列（VehicleId 是数字，不含 'ego'）
# Lane_ID 可能是 "5563;5564" 这种多值格式，用 str.contains 匹配任意目标车道
lane_pattern = '|'.join(all_target_lanes)
ego_mask = (
    (df['EntityType'] == 'ego') &
    (df['Lane_ID'].str.contains(lane_pattern, na=False))
)
target_times = df.loc[ego_mask, 'Time'].unique()

# 第二步：基于这些时间点，提取所有车辆（包括 ego 和周边 obj）的数据
final_df = df[df['Time'].isin(target_times)].copy()

# 5. 标记数据属于哪个交叉口（Lane_ID 可能含多值，取第一个匹配）
def mark_intersection(lane_str):
    for part in str(lane_str).split(';'):
        part = part.strip()
        for name, lanes in intersections.items():
            if part in lanes:
                return name
    return "Other"

final_df['Intersection_Group'] = final_df['Lane_ID'].apply(mark_intersection)

# 保存结果
final_df.to_csv('filtered_intersection_data_224_10.csv', index=False)
print(f"筛选完成，共提取到 {len(final_df)} 条数据，涵盖 {len(target_times)} 个时间帧。")
