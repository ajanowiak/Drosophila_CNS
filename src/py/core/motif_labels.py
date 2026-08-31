# motif_labels.py

"""
Loads motif ID -> name annotations and builds display labels for plots.

Pipeline context: shared across every stage that attaches a human-readable
motif name to a motif ID column (shap_importance, first_logit_model,
second_logit_model) -- lives in core/ specifically so it's importable from
any stage directory via PYTHONPATH=src/py.

Inputs: a motif annotation TSV with "id" and "name" columns.
Outputs: none (returns loaded/derived data in memory).
"""

import pandas as pd


def load_motif_annotations(path: str, sep: str) -> dict[str, str]:
    """Load the motif annotation table as a motif_id -> name dict."""
    annot = pd.read_csv(path, sep=sep)
    annot = annot.drop_duplicates("id").set_index("id")
    return annot["name"].astype(str).to_dict()


def motif_display_labels(motif_ids, id_to_name: dict[str, str]) -> list[str]:
    """
    Build "name  -  (id)" display labels for an iterable of motif IDs, in
    the given order.

    IDs with no matching annotation keep their raw motif ID.
    """
    labels = []
    for motif_id in motif_ids:
        name = id_to_name.get(motif_id)
        labels.append(f"{name}  -  ({motif_id})" if name is not None else motif_id)
    return labels
