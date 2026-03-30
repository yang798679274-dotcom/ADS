import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

BASE = "/Users/xinweiyang/Desktop/TTC_224/PET_Unprotect_left_turn"

# ── Read data ──────────────────────────────────────────────
df  = pd.read_csv(f"{BASE}/PET_Summary_ALL_Polygon.csv")
pet = df["AbsPET(s)"].dropna()
pet = pet[pet > 0].values

# ── Statistics variables ────────────────────────────────────────────────
n      = len(pet)
mean   = np.mean(pet)
median = np.median(pet)
sd     = np.std(pet, ddof=1)
cv     = sd / mean
skew   = stats.skew(pet)
kurt   = stats.kurtosis(pet)
pmin   = np.min(pet)
pmax   = np.max(pet)
p5     = np.percentile(pet, 5)
p25    = np.percentile(pet, 25)
p75    = np.percentile(pet, 75)
p95    = np.percentile(pet, 95)
iqr    = p75 - p25

g1 = df[df["Polygon"] == "overlap1_4"]["AbsPET(s)"].dropna().values
g2 = df[df["Polygon"] == "overlap2_4"]["AbsPET(s)"].dropna().values

BLUE   = "#2E86AB"
ORANGE = "#F18F01"
GREEN  = "#4CAF50"
RED    = "#E74C3C"
GRAY   = "#95A5A6"
DARK   = "#2C3E50"
BG     = "white"

# ════════════════════════════════════════════════════════════
# Figure (A) — Histogram + KDE
# ════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(9, 5), facecolor=BG)
ax1.set_facecolor(BG)

counts_h, edges = np.histogram(pet, bins=18)
for left, right, c in zip(edges[:-1], edges[1:], counts_h):
    ax1.bar(left, c, width=right - left, align="edge",
            color=BLUE, alpha=0.65, edgecolor="white", linewidth=0.8)

kde   = gaussian_kde(pet, bw_method=0.35)
x_kde = np.linspace(pmin * 0.8, pmax * 1.05, 400)
ax1_r = ax1.twinx()
ax1_r.plot(x_kde, kde(x_kde), color=DARK, lw=2.2, label="KDE")
ax1_r.set_ylabel("Density", fontsize=10, color=DARK)
ax1_r.tick_params(axis='y', labelcolor=DARK, labelsize=9)
ax1_r.set_ylim(0)
ax1_r.set_facecolor(BG)

ax1.axvline(mean,   color=ORANGE, lw=2,   linestyle="-",  label=f"Mean = {mean:.2f}s")
ax1.axvline(median, color=GREEN,  lw=2,   linestyle="--", label=f"Median = {median:.2f}s")
ax1.axvline(1.5,    color=RED,    lw=1.5, linestyle=":",  label="Critical (1.5s)")
ax1.axvline(5.0,    color=ORANGE, lw=1.2, linestyle=":",  label="Conflict (5.0s)")
ax1.axvspan(p25, p75, alpha=0.12, color=BLUE, label=f"IQR [{p25:.1f}, {p75:.1f}]s")

ax1.set_xlabel("PET (s)", fontsize=11)
ax1.set_ylabel("Count", fontsize=11)
ax1.set_title("Histogram + KDE\nPET — Unprotected Left Turn (N=57)",
              fontsize=11, fontweight="bold", color=DARK, pad=10)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right",
           framealpha=0.88, ncol=2)
ax1.grid(True, alpha=0.25, axis='y')

plt.tight_layout()
plt.savefig(f"{BASE}/Fig_A_Histogram_KDE.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ Fig_A_Histogram_KDE.png")

# ════════════════════════════════════════════════════════════
# Figure (B) — Descriptive Statistics Table
# ════════════════════════════════════════════════════════════
fig, ax2 = plt.subplots(figsize=(5, 7), facecolor=BG)
ax2.set_facecolor(BG)
ax2.set_axis_off()

stats_items = [
    ("N",          f"{n}"),
    ("Mean",       f"{mean:.3f} s"),
    ("Median",     f"{median:.3f} s"),
    ("SD",         f"{sd:.3f} s"),
    ("CV",         f"{cv:.3f}"),
    ("Skewness",   f"{skew:.3f}"),
    ("Kurtosis",   f"{kurt:.3f}"),
    ("Min",        f"{pmin:.3f} s"),
    ("P5",         f"{p5:.3f} s"),
    ("P25 (Q1)",   f"{p25:.3f} s"),
    ("P75 (Q3)",   f"{p75:.3f} s"),
    ("P95",        f"{p95:.3f} s"),
    ("Max",        f"{pmax:.3f} s"),
    ("IQR",        f"{iqr:.3f} s"),
]

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.text(0.5, 0.975, "(B) Descriptive Statistics",
         ha="center", va="top", fontsize=12, fontweight="bold", color=DARK)
ax2.text(0.5, 0.935, "PET — Unprotected Left Turn",
         ha="center", va="top", fontsize=9, color=GRAY)

row_h = 0.052
for i, (label, value) in enumerate(stats_items):
    y = 0.895 - i * row_h
    bg = "#EAF4FB" if i % 2 == 0 else "white"
    ax2.add_patch(FancyBboxPatch((0.04, y - 0.020), 0.92, row_h - 0.004,
                                  boxstyle="round,pad=0.003",
                                  facecolor=bg, edgecolor="#D5E8F3", linewidth=0.6, zorder=0))
    ax2.text(0.10, y + 0.006, label, ha="left", va="center", fontsize=10, color=DARK)
    ax2.text(0.90, y + 0.006, value, ha="right", va="center", fontsize=10,
             fontweight="bold", color=BLUE)

plt.tight_layout()
plt.savefig(f"{BASE}/Fig_B_Stats_Table.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ Fig_B_Stats_Table.png")

# ════════════════════════════════════════════════════════════
# Figure (C) — Box Plot
# ════════════════════════════════════════════════════════════
fig, ax3 = plt.subplots(figsize=(6, 5.5), facecolor=BG)
ax3.set_facecolor(BG)

data_box   = [pet, g1, g2]
labels_box = ["All\n(N=57)", "Conflict Zone 1\n(N=25)", "Conflict Zone 2\n(N=32)"]
colors_box = [BLUE, "#5DADE2", "#F39C12"]

bp = ax3.boxplot(data_box, labels=labels_box, patch_artist=True,
                 showfliers=True, widths=0.5,
                 medianprops=dict(color="white", lw=2.5),
                 whiskerprops=dict(lw=1.5),
                 capprops=dict(lw=2),
                 flierprops=dict(marker='o', markersize=4, alpha=0.5))
for patch, c in zip(bp['boxes'], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)

ax3.axhline(1.5, color=RED,    linestyle="--", lw=1.5, alpha=0.8, label="Critical threshold (1.5s)")
ax3.axhline(5.0, color=ORANGE, linestyle="--", lw=1.5, alpha=0.8, label="Conflict threshold (5.0s)")
ax3.set_ylabel("PET (s)", fontsize=11)
ax3.set_title("Box Plot\nPET — Unprotected Left Turn",
              fontsize=11, fontweight="bold", color=DARK, pad=10)
ax3.legend(fontsize=9, loc="upper right", framealpha=0.88)
ax3.grid(True, alpha=0.25, axis='y')

plt.tight_layout()
plt.savefig(f"{BASE}/Fig_C_BoxPlot.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ Fig_C_BoxPlot.png")

# ════════════════════════════════════════════════════════════
# Figure (D) — Empirical CDF
# ════════════════════════════════════════════════════════════
fig, ax4 = plt.subplots(figsize=(7, 5), facecolor=BG)
ax4.set_facecolor(BG)

sorted_pet = np.sort(pet)
ecdf = np.arange(1, n + 1) / n
ax4.plot(sorted_pet, ecdf, color=BLUE, lw=2.2, label="Empirical CDF")

for pct_val, x_val, col in [(0.05, p5, GRAY), (0.25, p25, GREEN),
                              (0.50, median, GREEN), (0.75, p75, GREEN),
                              (0.95, p95, GRAY)]:
    ax4.plot(x_val, pct_val, 'o', color=col, ms=8, zorder=5)
    offset_x = 0.8 if x_val < pmax * 0.75 else -4.5
    ax4.annotate(f"P{int(pct_val*100)} = {x_val:.1f}s",
                 xy=(x_val, pct_val),
                 xytext=(x_val + offset_x, pct_val - 0.07),
                 fontsize=8.5, color=DARK,
                 arrowprops=dict(arrowstyle="-", lw=0.8, color=DARK))

ax4.axvline(1.5, color=RED,    linestyle=":", lw=1.5, label="Serious (1.5s)")
ax4.axvline(3.0, color=ORANGE, linestyle=":", lw=1.5, label="Minor (3.0s)")
ax4.axhline(0.5, color=GRAY,   linestyle="--", lw=1, alpha=0.6, label="50th percentile")
ax4.set_xlabel("AbsPET (s)", fontsize=11)
ax4.set_ylabel("Cumulative Probability", fontsize=11)
ax4.set_title("(D) Empirical CDF\nPost-Encroachment Time — Unprotected Left Turn  |  N=57",
              fontsize=11, fontweight="bold", color=DARK, pad=10)
ax4.legend(fontsize=9, loc="lower right", framealpha=0.88)
ax4.grid(True, alpha=0.25)
ax4.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f"{BASE}/Fig_D_Empirical_CDF.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ Fig_D_Empirical_CDF.png")

# ════════════════════════════════════════════════════════════
# Figure (E) — Mean ± SD by Participant
# ════════════════════════════════════════════════════════════
fig, ax5 = plt.subplots(figsize=(7, 5), facecolor=BG)
ax5.set_facecolor(BG)

part_stats = df.groupby("Participant")["AbsPET(s)"].agg(
    Mean="mean", SD="std", N="count", Median="median"
).reset_index().sort_values("Participant")

y_pos = np.arange(len(part_stats))
ax5.barh(y_pos, part_stats["Mean"], xerr=part_stats["SD"],
         color=BLUE, alpha=0.65, height=0.55,
         error_kw=dict(ecolor=DARK, lw=1.5, capsize=5))
ax5.plot(part_stats["Median"], y_pos, 'D', color=ORANGE, ms=8, zorder=5, label="Median")

ax5.axvline(mean, color=DARK,   linestyle="--", lw=1.8, label=f"Overall mean ({mean:.1f}s)")
ax5.axvline(1.5,  color=RED,    linestyle=":",  lw=1.4, label="Critical (1.5s)")
ax5.axvline(5.0,  color=GREEN,  linestyle=":",  lw=1.4, label="Conflict (5.0s)")

DAY_MAP = {224: "Day 1", 230: "Day 2", 251: "Day 3", 253: "Day 4",
           263: "Day 5", 265: "Day 6", 267: "Day 7"}
ax5.set_yticks(y_pos)
ax5.set_yticklabels([DAY_MAP.get(int(p), str(p)) for p in part_stats["Participant"]], fontsize=10)
ax5.set_xlabel("PET (s)", fontsize=11)
ax5.set_title("Mean ± SD by Day  (◆ = Median)\n"
              "PET — Unprotected Left Turn",
              fontsize=11, fontweight="bold", color=DARK, pad=10)
ax5.legend(fontsize=9, loc="lower right", framealpha=0.88)
ax5.grid(True, alpha=0.25, axis='x')

plt.tight_layout()
plt.savefig(f"{BASE}/Fig_E_Mean_SD_Participant.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ Fig_E_Mean_SD_Participant.png")

# # ════════════════════════════════════════════════════════════
# # Figure (F) — Distribution Fitting
# # ════════════════════════════════════════════════════════════
# from scipy.stats import lognorm, weibull_min, gamma as gamma_dist, kstest

# fit_dists = [
#     ("Lognormal", lognorm,      "#E74C3C"),
#     ("Weibull",   weibull_min,  "#2ECC71"),
#     ("Gamma",     gamma_dist,   "#9B59B6"),
# ]

# fit_results = {}
# for name, dist, _ in fit_dists:
#     params = dist.fit(pet, floc=0)
#     ks_stat, p_val = kstest(pet, dist.name, args=params)
#     fit_results[name] = {"params": params, "KS": ks_stat, "p": p_val, "dist": dist}

# best = min(fit_results, key=lambda k: fit_results[k]["KS"])

# fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
# fig.suptitle("(F) Distribution Fitting — Post-Encroachment Time\n"
#              "Unprotected Left Turn  |  N=57",
#              fontsize=12, fontweight="bold", color=DARK, y=1.01)

# x_fit = np.linspace(pmin * 0.5, pmax * 1.05, 500)

# # ── Left: PDF + Histogram ──────────────────────────────────
# ax_pdf = axes[0]
# ax_pdf.set_facecolor(BG)

# counts_f, edges_f = np.histogram(pet, bins=18, density=True)
# for left, right, c in zip(edges_f[:-1], edges_f[1:], counts_f):
#     ax_pdf.bar(left, c, width=right - left, align="edge",
#                color=BLUE, alpha=0.35, edgecolor="white", linewidth=0.7)

# for name, dist, color in fit_dists:
#     params = fit_results[name]["params"]
#     ks     = fit_results[name]["KS"]
#     p      = fit_results[name]["p"]
#     lw     = 2.8 if name == best else 1.6
#     ls     = "-"  if name == best else "--"
#     label  = f"{name}  KS={ks:.3f}, p={p:.3f}{'  ★' if name == best else ''}"
#     ax_pdf.plot(x_fit, dist.pdf(x_fit, *params), color=color, lw=lw, ls=ls, label=label)

# ax_pdf.axvline(1.5, color=RED,    linestyle=":", lw=1.4, label="Critical (1.5s)")
# ax_pdf.axvline(5.0, color=ORANGE, linestyle=":", lw=1.4, label="Conflict (5.0s)")
# ax_pdf.set_xlabel("AbsPET (s)", fontsize=11)
# ax_pdf.set_ylabel("Density", fontsize=11)
# ax_pdf.set_title("PDF Fit vs Empirical Histogram", fontsize=11, fontweight="bold", color=DARK)
# ax_pdf.legend(fontsize=9, framealpha=0.88)
# ax_pdf.grid(True, alpha=0.22, axis='y')

# # ── Right: CDF comparison ──────────────────────────────────
# ax_cdf = axes[1]
# ax_cdf.set_facecolor(BG)

# sorted_pet = np.sort(pet)
# ecdf = np.arange(1, n + 1) / n
# ax_cdf.step(sorted_pet, ecdf, color=BLUE, lw=2.2, label="Empirical CDF", where="post")

# for name, dist, color in fit_dists:
#     params = fit_results[name]["params"]
#     lw     = 2.8 if name == best else 1.6
#     ls     = "-"  if name == best else "--"
#     ax_cdf.plot(x_fit, dist.cdf(x_fit, *params), color=color, lw=lw, ls=ls,
#                 label=f"{name}{'  ★' if name == best else ''}")

# ax_cdf.axvline(1.5, color=RED,    linestyle=":", lw=1.4)
# ax_cdf.axvline(5.0, color=ORANGE, linestyle=":", lw=1.4)
# ax_cdf.set_xlabel("AbsPET (s)", fontsize=11)
# ax_cdf.set_ylabel("Cumulative Probability", fontsize=11)
# ax_cdf.set_title("CDF Fit vs Empirical CDF", fontsize=11, fontweight="bold", color=DARK)
# ax_cdf.legend(fontsize=9, framealpha=0.88)
# ax_cdf.grid(True, alpha=0.22)
# ax_cdf.set_ylim(0, 1.05)

# # ── KS result annotation table ─────────────────────────────
# table_x, table_y = 0.03, 0.62
# ax_cdf.text(table_x, table_y, "KS Test Summary", transform=ax_cdf.transAxes,
#             fontsize=8.5, fontweight="bold", color=DARK,
#             bbox=dict(facecolor="white", edgecolor="#CCCCCC", boxstyle="round,pad=0.4"))
# for i, (name, _, color) in enumerate(fit_dists):
#     ks  = fit_results[name]["KS"]
#     p   = fit_results[name]["p"]
#     sig = "✓ accept" if p > 0.05 else "✗ reject"
#     ax_cdf.text(table_x, table_y - 0.09 * (i + 1),
#                 f"{name}: KS={ks:.3f}, p={p:.3f}  {sig}",
#                 transform=ax_cdf.transAxes, fontsize=8, color=color,
#                 fontweight="bold" if name == best else "normal")

# plt.tight_layout()
# plt.savefig(f"{BASE}/Fig_F_Distribution_Fitting.png", dpi=180, bbox_inches="tight", facecolor=BG)
# plt.close()
# print("✓ Fig_F_Distribution_Fitting.png")

# # ════════════════════════════════════════════════════════════
# # Figure (G) — Extreme Value Theory (EVT / POT / GPD)
# # ════════════════════════════════════════════════════════════
# from scipy.stats import genpareto

# # ── POT setup: lower tail (dangerous low PET) ─────────────
# # Reflect: exceedance Y = threshold - PET  for PET < threshold
# THRESHOLD = 5.0          # seconds — chosen from visual inspection
# below     = pet[pet < THRESHOLD]
# exceed    = THRESHOLD - below   # Y > 0: how far below threshold

# n_total  = len(pet)
# n_exceed = len(exceed)
# rate     = n_exceed / n_total   # proportion below threshold

# # Fit GPD to exceedances
# shape, loc_gpd, scale = genpareto.fit(exceed, floc=0)

# # ── Mean Excess values for MEP ─────────────────────────────
# thresholds = np.linspace(pet.min() + 0.1, THRESHOLD - 0.1, 40)
# me_vals, me_ci_lo, me_ci_hi = [], [], []
# for u in thresholds:
#     exc = THRESHOLD - pet[pet < u]
#     if len(exc) < 3:
#         continue
#     me  = exc.mean()
#     se  = exc.std(ddof=1) / np.sqrt(len(exc))
#     me_vals.append((u, me, me - 1.96 * se, me + 1.96 * se))
# me_arr = np.array(me_vals)

# # ── Return period grid ─────────────────────────────────────
# pet_grid = np.linspace(0.5, THRESHOLD - 0.01, 200)
# exceedance_prob, return_period = [], []
# for t in pet_grid:
#     y = THRESHOLD - t
#     p_tail = rate * (1 - genpareto.cdf(y, shape, loc=0, scale=scale))
#     exceedance_prob.append(p_tail)
#     return_period.append(1 / p_tail if p_tail > 0 else np.inf)

# exceedance_prob = np.array(exceedance_prob)
# return_period   = np.array(return_period)

# # Key risk estimates
# def pet_at_rp(rp_target):
#     idx = np.argmin(np.abs(return_period - rp_target))
#     return pet_grid[idx]

# rp5   = pet_at_rp(5)
# rp10  = pet_at_rp(10)
# rp20  = pet_at_rp(20)

# # ── Layout ─────────────────────────────────────────────────
# fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=BG)
# fig.suptitle("(G) Extreme Value Theory — Peaks Over Threshold (POT)\n"
#              "Post-Encroachment Time, Unprotected Left Turn  |  N=57  |  Threshold = 5.0s",
#              fontsize=12, fontweight="bold", color=DARK, y=1.01)

# # ── G1: Mean Excess Plot ───────────────────────────────────
# ax_mep = axes[0, 0]
# ax_mep.set_facecolor(BG)
# ax_mep.plot(me_arr[:, 0], me_arr[:, 1], color=BLUE, lw=2, marker='o', ms=4, label="Mean Excess")
# ax_mep.fill_between(me_arr[:, 0], me_arr[:, 2], me_arr[:, 3],
#                     alpha=0.2, color=BLUE, label="95% CI")
# ax_mep.axvline(THRESHOLD, color=RED, linestyle="--", lw=1.8, label=f"Threshold = {THRESHOLD}s")
# ax_mep.set_xlabel("Threshold u (s)", fontsize=10)
# ax_mep.set_ylabel("Mean Excess E[u − PET | PET < u]", fontsize=10)
# ax_mep.set_title("(i) Mean Excess Plot\n(linearity ≈ GPD valid)", fontsize=10,
#                  fontweight="bold", color=DARK)
# ax_mep.legend(fontsize=9, framealpha=0.88)
# ax_mep.grid(True, alpha=0.22)

# # ── G2: GPD fit to exceedances ─────────────────────────────
# ax_gpd = axes[0, 1]
# ax_gpd.set_facecolor(BG)

# exc_sorted = np.sort(exceed)
# exc_ecdf   = np.arange(1, n_exceed + 1) / n_exceed
# ax_gpd.step(exc_sorted, exc_ecdf, color=BLUE, lw=2, where="post", label="Empirical CDF")

# y_grid = np.linspace(0, exceed.max() * 1.1, 300)
# ax_gpd.plot(y_grid, genpareto.cdf(y_grid, shape, loc=0, scale=scale),
#             color=RED, lw=2.5, label=f"GPD fit\nξ={shape:.3f}, σ={scale:.3f}")

# # 95% CI via parametric bootstrap
# np.random.seed(42)
# boot_cdfs = []
# for _ in range(500):
#     sample = genpareto.rvs(shape, loc=0, scale=scale, size=n_exceed)
#     s, _, sc = genpareto.fit(sample, floc=0)
#     boot_cdfs.append(genpareto.cdf(y_grid, s, loc=0, scale=sc))
# boot_arr = np.array(boot_cdfs)
# ax_gpd.fill_between(y_grid, np.percentile(boot_arr, 2.5, axis=0),
#                     np.percentile(boot_arr, 97.5, axis=0),
#                     alpha=0.2, color=RED, label="95% Bootstrap CI")

# ax_gpd.set_xlabel("Exceedance Y = Threshold − PET (s)", fontsize=10)
# ax_gpd.set_ylabel("Cumulative Probability", fontsize=10)
# ax_gpd.set_title(f"(ii) GPD Fit to Exceedances\n(n_exceed={n_exceed}, rate={rate:.2f})",
#                  fontsize=10, fontweight="bold", color=DARK)
# ax_gpd.legend(fontsize=9, framealpha=0.88)
# ax_gpd.grid(True, alpha=0.22)

# # ── G3: Return Period curve ────────────────────────────────
# ax_rp = axes[1, 0]
# ax_rp.set_facecolor(BG)

# valid = return_period < 200
# ax_rp.plot(return_period[valid], pet_grid[valid], color="#9B59B6", lw=2.5,
#            label="GPD Return Level")
# ax_rp.axhline(1.5, color=RED,    linestyle="--", lw=1.5, label="Critical (1.5s)")
# ax_rp.axhline(3.0, color=ORANGE, linestyle="--", lw=1.5, label="Conflict (3.0s)")

# for rp_val, pet_val, col in [(5, rp5, "#E74C3C"), (10, rp10, "#F39C12"), (20, rp20, "#8E44AD")]:
#     ax_rp.plot(rp_val, pet_val, 'o', color=col, ms=9, zorder=5)
#     ax_rp.annotate(f"RP={rp_val}\nPET={pet_val:.2f}s",
#                    xy=(rp_val, pet_val), xytext=(rp_val + 5, pet_val + 0.3),
#                    fontsize=8, color=col,
#                    arrowprops=dict(arrowstyle="->", lw=1, color=col))

# ax_rp.set_xlabel("Return Period (interactions)", fontsize=10)
# ax_rp.set_ylabel("PET (s)", fontsize=10)
# ax_rp.set_title("(iii) Return Period Curve\n(expected PET level per N interactions)",
#                 fontsize=10, fontweight="bold", color=DARK)
# ax_rp.legend(fontsize=9, framealpha=0.88)
# ax_rp.grid(True, alpha=0.22)
# ax_rp.set_xlim(left=1)

# # ── G4: Summary table ─────────────────────────────────────
# ax_tbl = axes[1, 1]
# ax_tbl.set_facecolor(BG)
# ax_tbl.set_axis_off()
# ax_tbl.set_xlim(0, 1)
# ax_tbl.set_ylim(0, 1)

# ax_tbl.text(0.5, 0.97, "(iv) EVT Summary", ha="center", va="top",
#             fontsize=11, fontweight="bold", color=DARK)

# tbl_items = [
#     ("Method",              "Peaks Over Threshold (POT)"),
#     ("Threshold u",         f"{THRESHOLD:.1f} s"),
#     ("N total",             f"{n_total}"),
#     ("N below threshold",   f"{n_exceed}  ({rate*100:.1f}%)"),
#     ("GPD shape  ξ",        f"{shape:.4f}"),
#     ("GPD scale  σ",        f"{scale:.4f}"),
#     ("─────────────────", "─────────────────"),
#     ("PET at RP=5",         f"{rp5:.2f} s"),
#     ("PET at RP=10",        f"{rp10:.2f} s"),
#     ("PET at RP=20",        f"{rp20:.2f} s"),
#     ("─────────────────", "─────────────────"),
#     ("P(PET ≤ 1.5s)",       f"{np.interp(1.5, pet_grid, exceedance_prob)*100:.2f}%"),
#     ("P(PET ≤ 3.0s)",       f"{np.interp(3.0, pet_grid, exceedance_prob)*100:.2f}%"),
# ]

# row_h = 0.058
# for i, (label, value) in enumerate(tbl_items):
#     y = 0.90 - i * row_h
#     if "───" in label:
#         ax_tbl.axhline(y + row_h * 0.4, xmin=0.03, xmax=0.97,
#                        color="#CCCCCC", lw=0.8)
#         continue
#     bg = "#EAF4FB" if i % 2 == 0 else "white"
#     ax_tbl.add_patch(FancyBboxPatch((0.03, y - 0.020), 0.94, row_h - 0.006,
#                                      boxstyle="round,pad=0.002",
#                                      facecolor=bg, edgecolor="none"))
#     ax_tbl.text(0.07, y + 0.008, label, ha="left", va="center",
#                 fontsize=9, color=DARK)
#     ax_tbl.text(0.96, y + 0.008, value, ha="right", va="center",
#                 fontsize=9, fontweight="bold", color=BLUE)

# plt.tight_layout()
# plt.savefig(f"{BASE}/Fig_G_EVT_POT.png", dpi=180, bbox_inches="tight", facecolor=BG)
# plt.close()
# print("✓ Fig_G_EVT_POT.png")

print("\nAll 7 figures exported.")
