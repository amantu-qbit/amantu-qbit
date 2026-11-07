"""Summarise results/scores.csv into results/summary.json, results/summary.md and figure.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).parent
R = HERE / "results"
COND_X = {"truncated": -40, "original": 0, "pad25": 25, "pad50": 50, "pad100": 100}
EXTRA = ["repeat100"]  # control: +100% words from the essay's own sentences
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "D"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
rng = np.random.default_rng(0)


def boot_ci(x, n=5000):
    x = np.asarray(x)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def short(m):
    return m.split("/")[-1].replace("-Instruct", "").replace("Qwen2.5-", "Qwen2.5 ")


d = pd.read_csv(R / "scores.csv")
models = list(dict.fromkeys(d.model))
wide = d.pivot_table(index=["model", "prompt", "essay_id", "human"], columns="condition", values="score").reset_index()
wl = d[d.condition == "original"].drop_duplicates("essay_id")

summary = {"n_essays": int(wide.essay_id.nunique()),
           "human_vs_length_spearman": round(spearmanr(wl.human, wl.words)[0], 3), "graders": []}
for (m, p), g in wide.groupby(["model", "prompt"]):
    row = {"model": m, "prompt": p,
           "validity_spearman_vs_human": round(spearmanr(g.human, g.original)[0], 3),
           "mean_original": round(g.original.mean(), 3)}
    for c in ["pad25", "pad50", "pad100", "truncated"] + [c for c in EXTRA if c in g]:
        delta = g[c] - g.original
        mean, lo, hi = boot_ci(delta)
        row[c] = {"delta": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)],
                  "share_up": round(float((delta > 0).mean()), 3),
                  "wilcoxon_p": float(f"{wilcoxon(delta).pvalue:.2g}")}
    summary["graders"].append(row)
json.dump(summary, open(R / "summary.json", "w"), indent=1)

# Markdown table for the README.
lines = ["| Grader | Prompt | ρ vs human | Δ +25% filler | Δ +50% | Δ +100% | essays scored higher at +100% | Δ truncated (−40%) |",
         "|---|---|---|---|---|---|---|---|"]
for r in summary["graders"]:
    f = lambda k: f"{r[k]['delta']:+.2f} [{r[k]['ci95'][0]:+.2f}, {r[k]['ci95'][1]:+.2f}]"
    lines.append(f"| {short(r['model'])} | {r['prompt'].replace('_', '-')} | {r['validity_spearman_vs_human']:.2f} | "
                 f"{f('pad25')} | {f('pad50')} | {f('pad100')} | {r['pad100']['share_up']:.0%} | {f('truncated')} |")
(R / "summary.md").write_text("\n".join(lines) + "\n")

# Figure: (A) dose-response per grader, (B) who gains: delta at +100% by human score.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})

xs = list(COND_X.values())
for i, m in enumerate(models):
    for p, ls, alpha in [("standard", "-", 1.0), ("length_aware", "--", 0.75)]:
        g = wide[(wide.model == m) & (wide.prompt == p)]
        stats = [boot_ci(g[c] - g.original) for c in COND_X]
        mean = [s[0] for s in stats]
        a.fill_between(xs, [s[1] for s in stats], [s[2] for s in stats], color=COLORS[i], alpha=0.10 * alpha, lw=0)
        a.plot(xs, mean, ls=ls, color=COLORS[i], lw=2, marker=MARKERS[i], ms=7,
               markeredgecolor=SURF, markeredgewidth=1.5, alpha=alpha,
               label=f"{short(m)} · {'standard rubric' if p == 'standard' else '“ignore length” rubric'}")
        if p == "standard":
            a.annotate(short(m), (100, mean[-1]), xytext=(8, 0), textcoords="offset points",
                       va="center", color=INK, fontsize=9)
a.axhline(0, color=INK2, lw=1)
a.axvline(0, color=GRID, lw=1)
a.set_xticks(xs, ["truncated\n(−40% text)", "original", "+25%\nfiller", "+50%\nfiller", "+100%\nfiller"])
a.set_xlim(-50, 122)
a.set_ylabel("Mean change in grade vs. original (1–6 scale)")
a.set_title("A   Filler that adds no content raises the grade", loc="left", fontweight="bold")
a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
for s in ["top", "right"]:
    a.spines[s].set_visible(False)
a.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)

big = models[-1]
g = wide[(wide.model == big) & (wide.prompt == "standard")]
lv = sorted(g.human.unique())
for k, (p, off, col) in enumerate([("standard", -0.18, COLORS[len(models) - 1]), ("length_aware", 0.18, INK2)]):
    gg = wide[(wide.model == big) & (wide.prompt == p)]
    st = [boot_ci((gg[gg.human == h].pad100 - gg[gg.human == h].original)) for h in lv]
    b.bar(np.array(lv) + off, [s[0] for s in st], width=0.34, color=col, alpha=1 if k == 0 else 0.45,
          label="standard rubric" if p == "standard" else "“ignore length” rubric")
    b.errorbar(np.array(lv) + off, [s[0] for s in st], yerr=[[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]],
               fmt="none", ecolor=INK2, lw=1, capsize=0)
b.axhline(0, color=INK2, lw=1)
b.set_xticks(lv); b.set_xlabel("Human score of the essay")
b.set_ylabel("Grade change at +100% filler")
b.set_title(f"B   Who gains? ({short(big)})", loc="left", fontweight="bold")
b.grid(axis="y", color=GRID, lw=0.8); b.set_axisbelow(True)
for s in ["top", "right"]:
    b.spines[s].set_visible(False)
b.legend(frameon=False, fontsize=8)

n = summary["n_essays"]
fig.text(0.01, 0.01, f"{n} essays from the public AES 2.0 corpus (tasksource/AES2-essay-scoring), stratified by human score. "
         "Paired design: same essay, filler sentences inserted. Shaded / bars: 95% bootstrap CI.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(HERE / "figure.png", dpi=200)
print(open(R / "summary.md").read())
print(json.dumps({k: v for k, v in summary.items() if k != "graders"}))
