configfile: "config/config.yml"

include: "compute_motif_enrichment.smk"

# Must match the values of core.constants.FilteringMode.
FILTERING_MODES = ["unfiltered", "neural_labels", "refined_annotations"]


rule all:
    input:
        expand(
            "results/training_data/{filtering_mode}/hrs{window}/.done",
            filtering_mode=FILTERING_MODES,
            window=config["windows"],
        )
