# Extracts the binary loop-presence vector (ML training target) for one
# tissue and time window. Unlike compute_motif_enrichment, this does not
# depend on filtering mode computationally -- filtering mode only selects
# which per-window output directory the vector is written to, so the same
# vector ends up duplicated across the unfiltered/neural_labels/
# refined_annotations directories, next to the corresponding enrichment
# matrices (the explanatory variables).
rule extract_loop_presence_vectors:
    input:
        loops="data/long_and_short_range_loops_D_mel.tsv",

    output:
        "results/training_data/{filtering_mode}/hrs{window}/y_{tissue}.csv"

    log:
        "logs/extract_loop_presence_vectors/{filtering_mode}_hrs{window}_{tissue}.log"

    conda:
        "../../env/motif_enrichment.yaml"

    shell:
        """
        PYTHONPATH=src/py python src/py/00_prepare_data/extract_loop_presence_vectors.py \
            --window {wildcards.window} \
            --tissue {wildcards.tissue} \
            --filtering_mode {wildcards.filtering_mode} \
            --loops_path {input.loops} \
            --output_dir results/training_data/{wildcards.filtering_mode}/hrs{wildcards.window} \
            --log_path {log}
        """
