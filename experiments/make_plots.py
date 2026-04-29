"""Génération des figures (PNG) pour le rapport à partir des JSON
produits par les scripts exp1..exp5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domineering.stats import wilson_ci  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)


def _save(fig, name):
    out = os.path.join(FIG, name)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote", out)


def plot_round_robin():
    path = os.path.join(RES, "exp1_round_robin.json")
    if not os.path.exists(path):
        print("skip round_robin (missing)")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    agents = data["agents"]
    n = len(agents)
    mat = np.full((n, n), np.nan)
    for r in data["matches"]:
        i = agents.index(r["a"])
        j = agents.index(r["b"])
        mat[i, j] = r["a_winrate"]
        mat[j, i] = 1.0 - r["a_winrate"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.set_yticklabels(agents)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="taux de victoire (ligne vs colonne)")
    ax.set_title(f"Tournoi round-robin "
                 f"({data['params']['size']}x{data['params']['size']}, "
                 f"budget={data['params']['budget']}, "
                 f"{data['params']['games']} parties)")
    _save(fig, "fig_round_robin.png")

    # Classement Elo-like : moyenne du taux de victoire par agent
    avg = np.nanmean(mat, axis=1)
    order = np.argsort(-avg)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([agents[i] for i in order][::-1], avg[order][::-1], color="steelblue")
    ax.set_xlabel("taux de victoire moyen sur le round-robin")
    ax.set_xlim(0, 1)
    ax.set_title("Classement (moyenne sur le round-robin)")
    _save(fig, "fig_round_robin_ranking.png")


def plot_budget():
    path = os.path.join(RES, "exp2_budget.json")
    if not os.path.exists(path):
        print("skip budget")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    n_games = data["params"]["games"]
    for name, rows in data["results"].items():
        xs = np.array([r["budget"] for r in rows])
        ys = np.array([r["winrate"] for r in rows])
        cis = [wilson_ci(r["wins"], r["wins"] + r["losses"]) for r in rows]
        lo = np.array([c[1] for c in cis])
        hi = np.array([c[2] for c in cis])
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker="o",
                    capsize=3, label=name)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("budget de simulations")
    ax.set_ylabel(f"taux de victoire vs Flat-{data['params']['baseline_budget']}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"Performance vs budget (CI Wilson 95%, {n_games} parties)")
    ax.grid(alpha=0.3)
    _save(fig, "fig_budget.png")


def plot_heavy():
    path = os.path.join(RES, "exp3_heavy.json")
    if not os.path.exists(path):
        print("skip heavy")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["results"]
    labels = [r["a"] for r in rows]
    vals = np.array([r["winrate"] for r in rows])
    cis = [wilson_ci(r["wins"], r["wins"] + r["losses"]) for r in rows]
    lo = np.array([c[1] for c in cis])
    hi = np.array([c[2] for c in cis])
    colors = ["#1f77b4" if "biased" in l else "#ff7f0e" for l in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, vals, color=colors,
           yerr=[vals - lo, hi - vals], capsize=4, ecolor="black")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("taux de victoire vs version *random*")
    ax.set_title(f"Heavy playouts : softmax (bleu) vs $\\varepsilon$-greedy (orange)"
                 f" — CI Wilson 95% sur {data['params']['games']} parties")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, "fig_heavy.png")


def plot_uct_c():
    path = os.path.join(RES, "exp4_uct_c.json")
    if not os.path.exists(path):
        print("skip uct_c")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["results"]
    xs = np.array([r["c"] for r in rows])
    ys = np.array([r["winrate"] for r in rows])
    cis = [wilson_ci(r["wins"], r["wins"] + r["losses"]) for r in rows]
    lo = np.array([c[1] for c in cis])
    hi = np.array([c[2] for c in cis])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker="o", capsize=3)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("constante d'exploration $c$")
    ax.set_ylabel(f"taux de victoire vs Flat ({data['params']['games']} parties)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sensibilité de UCT à la constante d'exploration")
    ax.grid(alpha=0.3)
    _save(fig, "fig_uct_c.png")


def plot_size():
    path = os.path.join(RES, "exp5_size.json")
    if not os.path.exists(path):
        print("skip size")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["results"]
    sizes = [r["size"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(sizes, [r["winrate"] for r in rows], marker="o", color="green")
    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("taille du plateau")
    axes[0].set_ylabel("winrate GRAVE vs Flat")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Avantage de GRAVE selon la taille")

    axes[1].plot(sizes, [r["avg_time_grave"] for r in rows],
                 marker="o", label="GRAVE")
    axes[1].plot(sizes, [r["avg_time_flat"] for r in rows],
                 marker="s", label="Flat")
    axes[1].set_xlabel("taille du plateau")
    axes[1].set_ylabel("temps moyen par coup (s)")
    axes[1].set_yscale("log")
    axes[1].set_title("Coût computationnel")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    _save(fig, "fig_size.png")


def plot_levels():
    path = os.path.join(RES, "exp7_levels.json")
    if not os.path.exists(path):
        print("skip levels")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["results"]
    labels = [r["name"] for r in rows]
    ys = np.array([r["winrate"] for r in rows])
    lo = np.array([r["ci_lo"] for r in rows])
    hi = np.array([r["ci_hi"] for r in rows])
    colors = ["#5b9bd5" if l.startswith("NMCS") else "#ed7d31" for l in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, ys, color=colors,
           yerr=[ys - lo, hi - ys], capsize=4, ecolor="black")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"winrate vs Flat ({data['params']['games']} parties)")
    ax.set_title("NMCS et NRPA à différents niveaux")
    _save(fig, "fig_levels.png")


def plot_hyperparams():
    path = os.path.join(RES, "exp8_hyperparams.json")
    if not os.path.exists(path):
        print("skip hyperparams")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, xkey, xlabel, title in [
        (axes[0], "puct_c", "c", r"$c_{\mathrm{puct}}$", "PUCT"),
        (axes[1], "grave_refmin", "ref_min", r"$\mathrm{ref\_min}$ (GRAVE)", "GRAVE"),
        (axes[2], "biased_temp", "tau", r"température $\tau$", "UCT-biased"),
    ]:
        rows = data[key]
        xs = np.array([r[xkey] for r in rows])
        ys = np.array([r["winrate"] for r in rows])
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker="o", capsize=3)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("winrate vs Flat")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Ablation d'hyperparamètres "
                 f"({data['params']['games']} parties, CI Wilson 95%)")
    _save(fig, "fig_hyperparams.png")


def plot_6x6():
    path = os.path.join(RES, "exp9_6x6.json")
    if not os.path.exists(path):
        print("skip 6x6")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    agents = data["agents"]
    n = len(agents)
    mat = np.full((n, n), np.nan)
    for r in data["matches"]:
        i = agents.index(r["a"]); j = agents.index(r["b"])
        mat[i, j] = r["winrate"]; mat[j, i] = 1.0 - r["winrate"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.set_yticklabels(agents)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="taux de victoire (ligne vs colonne)")
    ax.set_title(f"Round-robin 6x6 (budget={data['params']['budget']}, "
                 f"{data['params']['games']} parties)")
    _save(fig, "fig_6x6.png")


def plot_multiseed():
    path = os.path.join(RES, "exp6_multiseed.json")
    if not os.path.exists(path):
        print("skip multiseed")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    agents = data["agents"]
    n = len(agents)
    # Per-agent average winrate with CI from aggregated matches
    per_agent_w = {a: 0 for a in agents}
    per_agent_n = {a: 0 for a in agents}
    for r in data["matches"]:
        per_agent_w[r["a"]] += r["wins_a"]
        per_agent_n[r["a"]] += r["n"]
        per_agent_w[r["b"]] += r["wins_b"]
        per_agent_n[r["b"]] += r["n"]
    items = []
    for a in agents:
        p, lo, hi = wilson_ci(per_agent_w[a], per_agent_n[a])
        items.append((a, p, lo, hi, per_agent_n[a]))
    items.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [x[0] for x in items]
    ps = np.array([x[1] for x in items])
    los = np.array([x[2] for x in items])
    his = np.array([x[3] for x in items])
    ns = items[0][4]
    ax.barh(names[::-1], ps[::-1],
            xerr=[(ps - los)[::-1], (his - ps)[::-1]], capsize=3,
            color="steelblue", ecolor="black")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"winrate moyen agrégé ({ns} parties / agent, CI Wilson 95%)")
    ax.set_title(f"Round-robin multi-seed 5x5 — {len(data['params']['seeds'])} seeds")
    _save(fig, "fig_multiseed.png")

    # Heatmap (average winrate matrix)
    mat = np.full((n, n), np.nan)
    for r in data["matches"]:
        i = agents.index(r["a"]); j = agents.index(r["b"])
        mat[i, j] = r["winrate"]; mat[j, i] = 1.0 - r["winrate"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.set_yticklabels(agents)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="taux de victoire (ligne vs colonne)")
    ax.set_title(f"Round-robin multi-seed 5x5 (matrice agrégée)")
    _save(fig, "fig_multiseed_matrix.png")


def plot_nrpa2p():
    path = os.path.join(RES, "exp13_nrpa2p.json")
    if not os.path.exists(path):
        print("skip nrpa2p")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["results"]
    labels = [f"{r['a']}\nvs {r['b']}" for r in rows]
    ys = np.array([r["winrate"] for r in rows])
    lo = np.array([r["ci_lo"] for r in rows])
    hi = np.array([r["ci_hi"] for r in rows])
    colors = []
    for r in rows:
        if r["b"] == "Flat":
            colors.append("#4c72b0")
        else:
            colors.append("#dd8452")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, ys, yerr=[ys - lo, hi - ys], capsize=4,
           color=colors, ecolor="black")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"winrate ({data['params']['games']} parties)")
    ax.set_title("NRPA-2P (deux politiques) "
                 "vs Flat (bleu) et NRPA classique (orange) - IC Wilson 95%")
    plt.setp(ax.get_xticklabels(), rotation=10, ha="right", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_nrpa2p.png")


def plot_features():
    path = os.path.join(RES, "exp12_features.json")
    if not os.path.exists(path):
        print("skip features")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [
        (axes[0], "vs_flat", "PPATCS x 3 feature sets vs Flat"),
        (axes[1], "vs_nrpa", f"PPATCS x 3 feature sets vs NRPA-L{data['params']['level']}"),
    ]:
        rows = data[key]
        labels = [r["feature_set"] for r in rows]
        ys = np.array([r["winrate"] for r in rows])
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        colors = ["#4c72b0", "#55a868", "#c44e52"]
        ax.bar(labels, ys, yerr=[ys - lo, hi - ys], capsize=4,
               color=colors, ecolor="black")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"winrate ({data['params']['games']} parties)")
        ax.set_title(title)
    fig.suptitle("Ablation des features PPATCS pour Domineering "
                 "(IC Wilson 95%)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_features.png")


def plot_extended():
    path = os.path.join(RES, "exp11_extended.json")
    if not os.path.exists(path):
        print("skip extended")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) extended vs Flat / NRPA
    rows = data["extended"]
    labels = [f"{r['a']}\nvs {r['b']}" for r in rows]
    ys = np.array([r["winrate"] for r in rows])
    lo = np.array([r["ci_lo"] for r in rows])
    hi = np.array([r["ci_hi"] for r in rows])
    colors = []
    for r in rows:
        if r["b"] == "Flat":
            colors.append("#1f77b4")
        else:
            colors.append("#9467bd")
    axes[0].bar(labels, ys, yerr=[ys - lo, hi - ys], capsize=4,
                color=colors, ecolor="black")
    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("winrate (IC Wilson 95%)")
    axes[0].set_title("Algos étendus vs Flat (bleu) / NRPA (violet)")
    plt.setp(axes[0].get_xticklabels(), rotation=15, ha="right", fontsize=8)

    # (b) misère + discounted
    rows_b = data["misere"] + data["discounted_nmcs"]
    labels_b = [f"{r['a']}\nvs {r['b']}" for r in rows_b]
    ysb = np.array([r["winrate"] for r in rows_b])
    lob = np.array([r["ci_lo"] for r in rows_b])
    hib = np.array([r["ci_hi"] for r in rows_b])
    axes[1].bar(labels_b, ysb, yerr=[ysb - lob, hib - ysb],
                capsize=4, color="#2ca02c", ecolor="black")
    axes[1].axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("winrate (IC Wilson 95%)")
    axes[1].set_title("Misère & playouts discountés")
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_extended.png")


def plot_ppatcs():
    path = os.path.join(RES, "exp10_ppatcs.json")
    if not os.path.exists(path):
        print("skip ppatcs")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["results"]
    labels = [f"{r['a']} vs {r['b']}" for r in rows]
    ys = np.array([r["winrate"] for r in rows])
    lo = np.array([r["ci_lo"] for r in rows])
    hi = np.array([r["ci_hi"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, ys, yerr=[ys - lo, hi - ys], capsize=4,
           color="#7e3ff2", ecolor="black")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"winrate ({data['params']['games']} parties, IC Wilson 95%)")
    ax.set_title("PPATCS (Playout Policy Adaptation with Move Features)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, "fig_ppatcs.png")


if __name__ == "__main__":
    plot_round_robin()
    plot_budget()
    plot_heavy()
    plot_uct_c()
    plot_size()
    plot_levels()
    plot_hyperparams()
    plot_6x6()
    plot_multiseed()
    plot_ppatcs()
    plot_extended()
    plot_features()
    plot_nrpa2p()
