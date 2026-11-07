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


NAMES = {"deepseek/deepseek-v4-flash": "DeepSeek V4 Flash", "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
         "google/gemma-4-31b-it": "Gemma 4 31B", "qwen/qwen3.7-plus": "Qwen3.7 Plus",
         "anthropic/claude-haiku-5.5": "Claude Haiku 5.5", "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
         "google/gemini-3.8-flash": "Gemini 3.8 Flash", "openai/gpt-5.6-luna": "GPT-5.6 Luna"}
DIGIT_ONLY = {"anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "google/gemini-3.8-flash", "openai/gpt-5.6-luna"}


def short(m):
    return NAMES.get(m) or m.split("/")[-1].replace("-Instruct", "").replace("Qwen2.5-", "Qwen2.5 ")


d = pd.concat([pd.read_csv(f) for f in sorted(R.glob("scores*.csv"))], ignore_index=True)
models = list(dict.fromkeys(d.model))
LOCAL = {m for m in models if m.startswith("Qwen/")}
wide = d.pivot_table(index=["model", "prompt", "essay_id", "human"], columns="condition", values="score").reset_index()
wl = d[d.condition == "original"].drop_duplicates("essay_id")

summary = {"n_essays": int(wide.essay_id.nunique()),
           "human_vs_length_spearman": round(spearmanr(wl.human, wl.words)[0], 3), "graders": []}
for (m, p), g in wide.groupby(["model", "prompt"]):
    row = {"model": m, "prompt": p,
           "validity_spearman_vs_human": round(spearmanr(g.human, g.original)[0], 3),
           "mean_original": round(g.original.mean(), 3)}
    for c in ["pad25", "pad50", "pad100", "truncated"] + EXTRA:
        if c not in g or g[c].isna().all():
            continue
        delta = (g[c] - g.original).dropna()
        mean, lo, hi = boot_ci(delta)
        row[c] = {"delta": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)],
                  "share_up": round(float((delta > 0).mean()), 3),
                  "wilcoxon_p": float(f"{wilcoxon(delta).pvalue:.2g}")}
    summary["graders"].append(row)
json.dump(summary, open(R / "summary.json", "w"), indent=1)

# Markdown table for the README.
lines = ["| Grader | Prompt | ρ vs human | Δ +25% filler | Δ +50% | Δ +100% | scored higher at +100% | Δ self-repeat +100% | Δ truncated −40% |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in summary["graders"]:
    f = lambda k: f"{r[k]['delta']:+.2f} [{r[k]['ci95'][0]:+.2f}, {r[k]['ci95'][1]:+.2f}]"
    lines.append(f"| {short(r['model'])}{' †' if r['model'] in DIGIT_ONLY else ''} | {r['prompt'].replace('_', '-')} | {r['validity_spearman_vs_human']:.2f} | "
                 f"{f('pad25')} | {f('pad50')} | {f('pad100')} | {r['pad100']['share_up']:.0%} | {f('repeat100') if 'repeat100' in r else 'n/a'} | {f('truncated')} |")
lines.append("\n† No log-probs available: score is the digit the model replies with at temperature 0, so per-essay "
             "scores are whole numbers and the means are noisier.")
(R / "summary.md").write_text("\n".join(lines) + "\n")

# Figure: (A) grader quality vs. what padding does to its grade, (B) dose-response for two graders.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
style = {True: (COLORS[0], "o", "open model, run locally on CPU"), False: (COLORS[1], "s", "hosted model, via OpenRouter")}
by = {(r["model"], r["prompt"]): r for r in summary["graders"]}
LABEL_RIGHT = {"google/gemma-4-31b-it", "google/gemini-3.8-flash", "Qwen/Qwen2.5-0.5B-Instruct",
               "Qwen/Qwen2.5-1.5B-Instruct"}
for m in models:
    r = by[(m, "standard")]
    col, mk, _ = style[m in LOCAL]
    x, y = r["validity_spearman_vs_human"], r["pad100"]["delta"]
    lo, hi = r["pad100"]["ci95"]
    hollow = m in DIGIT_ONLY
    a.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=mk, ms=9, color=col, ecolor=col, elinewidth=1.5,
               markerfacecolor=SURF if hollow else col, markeredgecolor=col if hollow else SURF,
               markeredgewidth=2 if hollow else 1.5, zorder=3)
    right = m in LABEL_RIGHT
    a.annotate(short(m) + (" †" if hollow else ""), (x, y), xytext=(10 if right else -10, 0),
               textcoords="offset points", fontsize=9, color=INK, va="center", ha="left" if right else "right")
for loc, (col, mk, lab) in style.items():
    a.plot([], [], mk, color=col, ms=8, label=lab)
a.plot([], [], "s", color=COLORS[1], markerfacecolor=SURF, markeredgewidth=2, ms=8,
       label="† hosted, no log-probs: temperature-0 digit reply")
a.set_xlim(0.1, 1.02)
a.axhline(0, color=INK2, lw=1)
a.set_xlabel("Grader agreement with human scores (Spearman ρ, original essays)")
a.set_ylabel("Grade change when filler doubles the essay (1–6 scale)")
a.set_title("A   Weak graders pay for filler; strong ones charge for it", loc="left", fontweight="bold")
a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
for sp in ["top", "right"]:
    a.spines[sp].set_visible(False)
a.legend(frameon=False, fontsize=8, loc="lower left")

pick = [m for m in ["Qwen/Qwen2.5-1.5B-Instruct", "deepseek/deepseek-v4-pro"] if m in models]
xs = list(COND_X.values())
for m in pick:
    col, mk, _ = style[m in LOCAL]
    for p, ls, alpha in [("standard", "-", 1.0), ("length_aware", "--", 0.6)]:
        g = wide[(wide.model == m) & (wide.prompt == p)]
        st = [boot_ci(g[c] - g.original) for c in COND_X]
        b.fill_between(xs, [s[1] for s in st], [s[2] for s in st], color=col, alpha=0.12 * alpha, lw=0)
        b.plot(xs, [s[0] for s in st], ls=ls, color=col, lw=2, marker=mk, ms=7, markeredgecolor=SURF,
               markeredgewidth=1.5, alpha=alpha,
               label=f"{short(m)} · {'standard rubric' if p == 'standard' else '“length is not a criterion” rubric'}")
b.axhline(0, color=INK2, lw=1)
b.set_xticks(xs, ["truncated\n(−40%)", "original", "+25%\nfiller", "+50%\nfiller", "+100%\nfiller"])
b.set_ylabel("Mean grade change vs. original")
b.set_title("B   Same essay, more words, no new content", loc="left", fontweight="bold")
b.grid(axis="y", color=GRID, lw=0.8); b.set_axisbelow(True)
for sp in ["top", "right"]:
    b.spines[sp].set_visible(False)
b.legend(frameon=False, fontsize=8, loc="lower left")

n = summary["n_essays"]
fig.text(0.01, 0.012, f"{n} essays from the public AES 2.0 corpus (tasksource/AES2-essay-scoring), 20 per human score 1–6. "
         "Paired design: each essay graded as written and with generic filler sentences inserted.\n"
         "Score = expected value of the grader's next-token distribution over 1–6 († graders: the digit replied at temperature 0). Error bars / bands: 95% bootstrap CI.",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(HERE / "figure.png", dpi=200)
print(open(R / "summary.md").read())
print(json.dumps({k: v for k, v in summary.items() if k != "graders"}))
