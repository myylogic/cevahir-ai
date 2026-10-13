"""Standalone descriptive figure generated from recorded primary results."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import relation_learning as r


def main():
    data = json.loads((r.RESULTS / "relation_learning.json").read_text(encoding="utf-8"))
    methods = (("table", "Yalnız hedef gözlemleri", "#777777", "--"),
               ("permutation_prior", "Hazır bijeksiyon önbilgisi", "#267eab", "-"),
               ("source_both", "İki öğrenilmiş kaynak + düzeltme", "#008461", "-"),
               ("frozen_source_both", "İki kaynak + çelişkiyi reddetme", "#c75528", ":"))
    titles = ("Kaynak ilişkileri geçerli", "İlişki ailesi değişmiş", "Tek geçişte istisna")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure, axes = plt.subplots(1, 3, figsize=(13, 4.9), sharey=True)
        for axis, regime, title in zip(axes, r.REGIMES, titles):
            for method, label, color, style in methods:
                rows = [row for row in data["runs"] if row["regime"] == regime and row["method"] == method]
                values = np.array([[s["composition"]["brier"] for s in row["snapshots"]] for row in rows])
                axis.plot(range(13), values.mean(axis=0), color=color, linestyle=style, label=label, linewidth=2.1)
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel("Gözlenmiş hedef geçişi")
            axis.set_xticks(range(0, 13, 2))
            axis.set_xlim(0, 12)
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Öğretilmemiş bileşimlerde Brier hatası\n(düşük daha iyi)")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, .04), frameon=False)
        figure.suptitle("Ayrı deneyimlerde öğrenilen ilişkilerin yeni hedefe aktarımı", fontsize=15, y=.99)
        figure.text(.5, .008, "24 yaşamın ortalaması · 6 durumlu sentetik ortam · Bileşim kuralı verilmiş · İnsan veya genel yapay zekâ deneyi değildir",
                    ha="center", fontsize=8.5, color="#555555")
        figure.subplots_adjust(left=.075, right=.985, top=.83, bottom=.27, wspace=.1)
        figure.savefig(r.RESULTS / "composition_curves.png", dpi=180)
        figure.savefig(r.RESULTS / "composition_curves.svg")
        plt.close(figure)
    print("Wrote composition_curves.png and composition_curves.svg")


if __name__ == "__main__":
    main()
