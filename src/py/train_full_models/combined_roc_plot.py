# combined_roc_plot.py

"""
Overlays the three tissues' ROC curves from train_time_specific.py into one
combined figure, for a single (model, window).

Pipeline context: reads the CVResult pickled by train_time_specific.py for
each tissue and re-plots them together - it does not re-run CV. Looping
over tissues here is intentional: a single overlay figure needs all three
tissues in the same process, the same exception already made for
compare_bar_plots.py.

Inputs:
  - results/time_specific/<model>/hrs<window>/roc_result_<model>_hrs<window>_<tissue>.pkl

Outputs:
  - results/figures/time_specific/<model>/hrs<window>/roc_<model>_combined_hrs<window>.{png,pdf}
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from core.constants import MODELS, TISSUES, WINDOWS
from core.log import configure_logging
from train_full_models.io import load_cv_result

logger = logging.getLogger(__name__)


def plot_combined_roc(model: str, window: str) -> None:
    """Overlay the per-tissue ROC curves for one (model, window) into one figure."""
    full = MODELS[model]["full"]

    fig, ax = plt.subplots(figsize=(6, 6))

    for tissue in TISSUES:
        result_path = (
            f"results/time_specific/{model}/hrs{window}/"
            f"roc_result_{model}_hrs{window}_{tissue}.pkl"
        )
        result = load_cv_result(result_path)

        ax.plot(result.mean_fpr, result.mean_tpr, label=tissue)
        ax.fill_between(result.mean_fpr, result.tprs_lower, result.tprs_upper, alpha=0.2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title=f"{full} ROC curves — hrs{window}")
    ax.legend()

    fig_dir = Path(f"results/figures/time_specific/{model}/hrs{window}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"roc_{model}_combined_hrs{window}.{fmt}", dpi=300, bbox_inches="tight")

    plt.close(fig)
    logger.info(f"Saved combined ROC plot to {fig_dir}/roc_{model}_combined_hrs{window}.{{png,pdf}}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay per-tissue time-specific ROC curves for one model and window"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--window", required=True, choices=WINDOWS)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)
    plot_combined_roc(args.model, args.window)
    logger.info("Done.")


if __name__ == "__main__":
    main()
