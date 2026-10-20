# human_prepare_data/constants.py

"""Shared constants for the human-brain loop-prediction prototype."""

# The three Rahman et al. 2023 Hi-C conditions, in a fixed order.
CONDITIONS = ["fetal", "glia", "neuron"]

# How a condition's negative loops are drawn (see build_loop_universe.py):
#   other_conditions - the other conditions' loops whose Hi-C rectangle coincides
#                      with no real loop of this condition (disjoint rectangles);
#   n2               - Originally called "new_pairs_of_anchors": random re-pairing of this
#                      condition's own loop anchors.
NEGATIVE_MODES = ["other_conditions", "n2"]

# Columns pulled from the Li 2023 atlas metatable (Table S3).
ATLAS_COLUMNS = ["barcode", "sample", "cellclass", "subclass", "celltype"]

# Atlas label given to nuclei absent from the atlas metatable.
UNANNOTATED = "unannotated"

# Autosomes plus chrX; chromVAR ignores everything else (chrY, contigs, chrM).
MAIN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

# Group name used when a feature set has no cell partition (one group of all
# analysis cells), and the separator in "<motif>__<group>" feature names.
POOLED_GROUP = "all"
GROUP_SEPARATOR = "__"

# Models that accept NaN features natively, so a loop with no both-open cell
# keeps its (NaN) enrichment row instead of being dropped.
NAN_TOLERANT_MODELS = {"XGB"}
