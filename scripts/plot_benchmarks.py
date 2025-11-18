import matplotlib.pyplot as plt
import numpy as np
import json 
from typing import Iterable, Dict, List
import os 
import matplotlib.patches as mpatches
import sys

def plot_benchmarks_as_violin_plots(versions: Iterable[str]):
    """Plots benchmark results as violin plots from different versions of the Xion solver. using JSON files stored in the benchmarks/ directory."""
    assert len(versions) <= 5, "Too many versions to plot at once, please plot at most 4 versions simultaneously."

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f"XION Benchmark Results Across Versions")
    ax.set_ylabel("log(Time in seconds)")
    ax.set_yscale("log")
    ax.grid(axis="y", linewidth=0.7, color="black", zorder=0, alpha=0.1)
    fig.tight_layout()

    version_colors = ["tab:green", "tab:red", "tab:blue", "tab:orange", "tab:purple"]
    widths = 0.8 / (2 * len(versions))
    for idx, (version, color) in enumerate(zip(versions, version_colors)):
        # Load benchmark results
        with open(os.path.join(os.getcwd(), "logs", "benchmarks", f"xion{version}.json"), "r") as file:
            times: Dict[str, List[float]] = json.load(file)

        # Make violin plots center around each problem
        positions = np.arange(1, len(times.keys()) + 1) + (idx - (len(versions) - 1) / 2) * 1.3 * widths
        vp = ax.violinplot([times[problem_type] for problem_type in times.keys()], 
                           positions=positions, 
                           widths = widths, showmedians=True)

        for violin in vp['bodies']:
            violin.set_facecolor(color)
            violin.set_edgecolor("black")
            violin.set_alpha(0.7)
            violin.set_zorder(3)

        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp[partname].set_edgecolor("black")
            vp[partname].set_linewidth(1)
    
    ax.set_xticks(ticks=np.arange(1, len(times.keys()) + 1), labels=list(times.keys()), rotation=45, ha="right")
    patches = [mpatches.Patch(color=color, label=f"XION {version}") for version, color in zip(versions, ["tab:green", "tab:red", "tab:blue", "tab:orange", "tab:purple"])]
    plt.legend(handles=patches, title="Versions")
    plt.show()

if __name__ == "__main__":
    versions_to_plot = sys.argv[1:]  # Pass versions as command-line arguments
    plot_benchmarks_as_violin_plots(versions_to_plot)