#!/usr/bin/env python3
# Plots the bounds.

import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from matplotlib.ticker import MaxNLocator

output_dir = "bounds"


def plot_bounds(bound_file, output_file):
    """Plots bounds from one file."""
    b = pandas.read_csv(bound_file)

    # Treat 'Num. levels' as a categorical variable for sensible hue coloring
    if "Num. levels" in b.columns:
        b["Num. levels"] = b["Num. levels"].astype("category")

    plt.figure(figsize=(6, 4.0))
    g = sns.lineplot(
        data=b, x="Num. cliques", y="Min. gates", hue="Num. levels", alpha=0.85, lw=1
    )

    # emphasize the line for what was averaged
    # pdb.set_trace()
    # max_cliques = b['Num. cliques'].max()
    # levels = b['Num. levels'].max()
    # b_avg = b[(b['Num. cliques'] == max_cliques) & (b['Num. levels'] == levels)]
    # plt.plot(b_avg['Num. cliqfues'], b_avg['Min. gates'], 'k-', lw=2)

    plt.xlabel("Number of cliques")
    plt.ylabel("Number of NAND gates")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


os.makedirs(output_dir, exist_ok=True)

import glob

for bound_file in glob.glob("bounds/*.csv"):
    base_name = os.path.splitext(os.path.basename(bound_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.pdf")
    plot_bounds(bound_file, output_file)
