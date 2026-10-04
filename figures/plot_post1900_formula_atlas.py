"""Render one normalized record from every post-1900 source-law family."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "post1900_formula_registry.json"
ARCHIVE = ROOT / "data" / "post1900_formula_pointclouds.npz"
OUT_PNG = ROOT / "figures" / "figure_post1900_formula_atlas.png"
OUT_PDF = ROOT / "figures" / "figure_post1900_formula_atlas.pdf"
COLORS = {
    "declared_dictionary_overlap": "#7F8C8D",
    "smooth_composite": "#0072B2",
    "threshold_or_piecewise": "#D55E00",
    "special_function_or_integral": "#009E73",
    "nonmonotone_or_oscillatory": "#CC79A7",
}


def main() -> None:
    registry = sorted(
        json.loads(REGISTRY.read_text(encoding="utf-8"))["laws"],
        key=lambda row: (row["first_valid_year"], row["family_id"]),
    )
    archive = np.load(ARCHIVE, allow_pickle=True)
    families = np.asarray(archive["family_ids"]).astype(str)
    clouds = archive["X"]
    ncols = 5
    nrows = int(np.ceil(len(registry)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 20), facecolor="#FBFBF9")
    for ax, row in zip(axes.flat, registry):
        cloud = clouds[np.flatnonzero(families == row["family_id"])[0]]
        x, y = cloud[:, 0], cloud[:, 1]
        xn = (x-x.min())/(max(float(np.ptp(x)), 1e-12))
        lo, hi = np.quantile(y, [0.01, 0.99])
        yn = (np.clip(y, lo, hi)-lo)/(max(float(hi-lo), 1e-12))
        color = COLORS[row["shape_class"]]
        ax.plot(xn, yn, color=color, linewidth=1.2)
        ax.scatter(xn[::6], yn[::6], color=color, s=4, alpha=0.65)
        title = row["family_id"].replace("_", " ")
        if len(title) > 31:
            title = title[:29] + "..."
        ax.set_title(f"{row['first_valid_year']}  {title}", fontsize=7, loc="left", pad=2)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#D8D8D8"); spine.set_linewidth(0.5)
        ax.set_facecolor("#FFFFFF")
    for ax in axes.flat[len(registry):]:
        ax.axis("off")
    fig.suptitle("Post-1900 formula benchmark atlas", x=0.05, y=0.995,
                 ha="left", fontsize=15, fontweight="bold")
    fig.text(0.05, 0.982,
             "One of eight generated records per source-law family; axes are independently normalized for shape audit.",
             ha="left", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.967, bottom=0.025, hspace=0.55, wspace=0.22)
    fig.savefig(OUT_PNG, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(json.dumps({"families": len(registry), "png": str(OUT_PNG), "pdf": str(OUT_PDF)}, indent=2))


if __name__ == "__main__":
    main()
