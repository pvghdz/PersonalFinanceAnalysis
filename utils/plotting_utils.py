import pandas as pd
import scipy.stats as stats
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def plot_expenses_with_outliers(
    expense_matrix: pd.DataFrame,
    outlier_mask: pd.DataFrame,
    save_dir: str,
    show: bool = False,
):
    from pathlib import Path
    from datetime import datetime

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_dir) / f"expenses_with_outliers.png"

    values = expense_matrix.values

    fig, ax = plt.subplots(figsize=(12, 6))

    # Base heatmap (values)
    im = ax.imshow(values, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Expense amount")

    # Overlay outliers as red X
    outlier_positions = np.where(outlier_mask.values)
    ax.scatter(
        outlier_positions[1],  # x = columns
        outlier_positions[0],  # y = rows
        marker="x",
        c='red',
        s=80,
        linewidths=2,
        zorder=3,
    )

    ax.set_xticks(range(len(expense_matrix.columns)))
    ax.set_xticklabels(expense_matrix.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(expense_matrix.index)))
    ax.set_yticklabels(expense_matrix.index)

    ax.set_title("Expense Matrix with Outliers Highlighted")
    ax.set_xlabel("Month")
    ax.set_ylabel("Category")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)

    logger.info(f"Expense + outlier overlay plot saved to: {filepath}")

    if show:
        plt.show()

    return filepath

def plot_outlier_mask(outlier_mask: pd.DataFrame, threshold: float, save_dir: str, show: bool = False):
    """
    Plots the outlier mask for a graphical display of which monthly values are considered outliers.

    """

    from pathlib import Path
    from datetime import datetime
    from matplotlib.colors import ListedColormap, BoundaryNorm

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{save_dir}/outlier_mask.png"

    #cmap = ListedColormap(["#f0f0f0", "#d62728"])
    cmap = ListedColormap(["#069AF3", "#FF6347"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(outlier_mask.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(outlier_mask.columns)))
    ax.set_xticklabels(outlier_mask.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(outlier_mask.index)))
    ax.set_yticklabels(outlier_mask.index)

    ax.set_title(f"Outlier Mask (Threshold: MAD > {threshold})")
    ax.set_xlabel("Category")
    ax.set_ylabel("Month")

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(["Normal", "Outlier"])

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)

    logger.info(f"Outlier mask plot saved to: {filepath}")

    if show:
        plt.show()

    return filepath