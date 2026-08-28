configfile: "config/config.yml"

include: "compute_motif_enrichment.smk"
include: "extract_loop_presence_vectors.smk"

# Must match the values of core.constants.FilteringMode.
FILTERING_MODES = ["unfiltered", "neural_labels", "refined_annotations"]


rule all:
    input:
        expand(
            "results/training_data/{filtering_mode}/hrs{window}/.done",
            filtering_mode=FILTERING_MODES,
            window=config["windows"],
        ),
        expand(
            "results/training_data/{filtering_mode}/hrs{window}/y_{tissue}.csv",
            filtering_mode=FILTERING_MODES,
            window=config["windows"],
            tissue=config["tissues"],
        ),
