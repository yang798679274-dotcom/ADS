import pandas as pd
import os
import glob
import requests
import json

URL = "http://70.229.15.100:9080/txvapi/history/getHistoryData"
input_dir = "224_signal_only_intersection_data_no_object"
output_dir = "224_spat_data"
os.makedirs(output_dir, exist_ok=True)


def fetch_spat(start_time, end_time):
    payload = {
        "DataType": "J2735Message",
        "FilterFlags": 2,
        "TimeFilter": {"StartTime": start_time, "EndTime": end_time}
    }
    resp = requests.post(URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_spat(response):
    rows = []
    for record in response.get("Response", []):
        timestamp = record["Time"]
        events = record.get("Events", [])
        event = ",".join(str(e) for e in events) if events else ""

        try:
            outer = json.loads(record["Json"])
            inner = json.loads(outer["Message"])

            if inner.get("MessageType") != "Spat":
                continue

            intersection_state = inner["Message"]["intersections"]["IntersectionState"]
            inter_id = intersection_state["id"]["id"]

            movement_states = intersection_state["states"]["MovementState"]
            if isinstance(movement_states, dict):
                movement_states = [movement_states]

            for ms in movement_states:
                signal_group = ms["signalGroup"]
                event_state = ms["state-time-speed"]["MovementEvent"]["eventState"]
                status = list(event_state.keys())[0]

                rows.append({
                    "Event": event,
                    "Timestamp": timestamp,
                    "IntersectionID": inter_id,
                    "SignalGroup": signal_group,
                    "Status": status
                })
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return rows


csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
print(f"Found {len(csv_files)} files\n")

for input_path in csv_files:
    filename = os.path.basename(input_path)
    stem = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{stem}_spat.csv")

    df = pd.read_csv(input_path)

    intersections = sorted(
        [g for g in df["Intersection_Group"].unique() if g.startswith("Intersection_")],
        key=lambda x: int(x.split("_")[1])
    )

    all_rows = []
    for inter in intersections:
        times = df.loc[df["Intersection_Group"] == inter, "Time"]
        start_time = times.iloc[0]
        end_time = times.iloc[-1]
        print(f"  {inter}: {start_time}  ->  {end_time} ... ", end="", flush=True)

        try:
            response = fetch_spat(start_time, end_time)
            rows = parse_spat(response)
            # tag each row with the intersection group
            for r in rows:
                r["Intersection_Group"] = inter
            all_rows.extend(rows)
            print(f"{len(rows)} records")
        except Exception as e:
            print(f"ERROR: {e}")

    if all_rows:
        out_df = pd.DataFrame(all_rows, columns=["Event", "Timestamp", "IntersectionID", "SignalGroup", "Status", "Intersection_Group"])
        out_df.to_csv(output_path, index=False)
        print(f"  -> Saved {len(all_rows)} rows to {output_path}\n")
    else:
        print(f"  -> No data for {filename}\n")

print("Done.")
