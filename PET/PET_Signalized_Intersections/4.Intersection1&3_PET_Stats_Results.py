import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import mannwhitneyu
import os

BASE_DIR = "/Users/xinweiyang/Desktop/TTC_224/PET_Unprotect_left_turn/"

df = pd.read_csv(os.path.join(BASE_DIR, "Intersection_1_3_PET_Summary_All.csv"))

inter1 = df[df["Intersection"] == "Intersection 1"]["PET(s)"].dropna().values
inter3 = df[df["Intersection"] == "Intersection 3"]["PET(s)"].dropna().values

COLORS = {"Intersection 1": "#2196F3", "Intersection 3": "#FF5722"}

# ============================================================
# Figure 1: Histogram + KDE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle("Histogram + KDE\nPET - Signalized Intersection Right Turn", fontsize=14, fontweight="bold")

for ax, data, label, color in zip(
    axes,
    [inter1, inter3],
    ["Intersection 1", "Intersection 3"],
    [COLORS["Intersection 1"], COLORS["Intersection 3"]],
):
    ax.hist(data, bins=10, density=True, alpha=0.55, color=color, edgecolor="white", label="Histogram")

    kde_x = np.linspace(data.min() - 1, data.max() + 1, 300)
    kde = stats.gaussian_kde(data, bw_method="scott")
    ax.plot(kde_x, kde(kde_x), color=color, linewidth=2.5, label="KDE")

    ax.axvline(np.mean(data), color="black", linestyle="--", linewidth=1.2, label=f"Mean = {np.mean(data):.2f}s")
    ax.axvline(np.median(data), color="gray", linestyle=":", linewidth=1.2, label=f"Median = {np.median(data):.2f}s")

    # IQR shaded region
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    ax.axvspan(q1, q3, alpha=0.15, color=color, label=f"IQR [{q1:.2f}, {q3:.2f}]s")

    # Threshold lines
    ax.axvline(1.5, color="#E53935", linestyle="-.", linewidth=1.5, label="Critical (1.5s)")
    ax.axvline(3.0, color="#FB8C00", linestyle="-.", linewidth=1.5, label="Conflict (3.0s)")

    panel_label = "(a)" if label == "Intersection 1" else "(b)"
    ax.set_title(f"{panel_label} {label} (N = {len(data)})", fontsize=12, fontweight="bold")
    ax.set_xlabel("PET (s)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(True)

plt.tight_layout()
hist_path = os.path.join(BASE_DIR, "PET_Histogram_KDE.png")
plt.savefig(hist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {hist_path}")

# ============================================================
# Figure 2: Box Plot
# ============================================================
overall = np.concatenate([inter1, inter3])
COLORS["Overall"] = "#4CAF50"

fig, ax = plt.subplots(figsize=(9, 6))

box_data   = [inter1, inter3, overall]
box_labels = ["Intersection 1", "Intersection 3", "Overall"]
box_colors = [COLORS["Intersection 1"], COLORS["Intersection 3"], COLORS["Overall"]]

bp = ax.boxplot(
    box_data,
    tick_labels=box_labels,
    patch_artist=True,
    notch=False,
    widths=0.45,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker="o", markersize=5, linestyle="none"),
)

for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Overlay individual data points (jittered)
for i, (data, color) in enumerate(zip(box_data, box_colors), start=1):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(data))
    ax.scatter(i + jitter, data, color=color, alpha=0.7, s=30, zorder=3)

ax.set_title("Box Plot\nPET - Signalized Intersection Right Turn", fontsize=13, fontweight="bold")
ax.set_ylabel("PET (s)", fontsize=11)
ax.set_xlabel("Intersection", fontsize=11)
for spine in ax.spines.values():
    spine.set_visible(True)

plt.tight_layout()
box_path = os.path.join(BASE_DIR, "PET_Boxplot.png")
plt.savefig(box_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {box_path}")

# ============================================================
# Figure 3: Stats Table
# ============================================================
def compute_stats(data, label):
    u_stat, p_val = mannwhitneyu(inter1, inter3, alternative="two-sided")
    _, p_norm = stats.shapiro(data)
    return {
        "Intersection": label,
        "N": len(data),
        "Mean (s)": round(np.mean(data), 3),
        "Median (s)": round(np.median(data), 3),
        "Std (s)": round(np.std(data, ddof=1), 3),
        "Min (s)": round(np.min(data), 3),
        "Max (s)": round(np.max(data), 3),
        "Q1 (s)": round(np.percentile(data, 25), 3),
        "Q3 (s)": round(np.percentile(data, 75), 3),
        "IQR (s)": round(np.percentile(data, 75) - np.percentile(data, 25), 3),
        "Shapiro p": round(p_norm, 4),
        "Mann-Whitney p": round(p_val, 4),
    }

rows = [compute_stats(inter1, "Intersection 1"), compute_stats(inter3, "Intersection 3")]
stats_df = pd.DataFrame(rows)

# Save CSV
stats_csv = os.path.join(BASE_DIR, "PET_Stats_Table.csv")
stats_df.to_csv(stats_csv, index=False)
print(f"Saved: {stats_csv}")

# Plot as table figure
fig, ax = plt.subplots(figsize=(14, 2.8))
ax.axis("off")

display_cols = [c for c in stats_df.columns if c != "Intersection"]
cell_text = stats_df[display_cols].values.tolist()
row_labels = stats_df["Intersection"].tolist()

tbl = ax.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=display_cols,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.7)

# Header styling
for j, _ in enumerate(display_cols):
    tbl[(0, j)].set_facecolor("#37474F")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# Row styling
row_colors = [COLORS["Intersection 1"], COLORS["Intersection 3"]]
for i, color in enumerate(row_colors, start=1):
    tbl[(i, -1)].set_facecolor(color)
    tbl[(i, -1)].set_text_props(color="white", fontweight="bold")
    for j in range(len(display_cols)):
        tbl[(i, j)].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")

ax.set_title("PET Descriptive Statistics", fontsize=13, fontweight="bold", pad=12)

plt.tight_layout()
table_path = os.path.join(BASE_DIR, "PET_Stats_Table.png")
plt.savefig(table_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {table_path}")

print("\n====== Stats Table ======")
print(stats_df.to_string(index=False))
