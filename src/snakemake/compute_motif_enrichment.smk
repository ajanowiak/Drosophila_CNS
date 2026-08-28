# Computes per-loop motif enrichment for one time window and one cell
# grouping strategy. The number of output files per group is only known
# once cell metadata is read (e.g. one enrichment matrix per tissue for
# "refined_annotations"), so completion is tracked with a sentinel file
# rather than listing every output explicitly.
rule compute_motif_enrichment:
    input:
        loops="data/new_time/hrs{window}_NNv1_time_matrix_loops.tsv",
        motifs="data/new_time/hrs{window}_NNv1_time_matrix_motifs.tsv",
        metadata="data/atac_meta.rds",

    output:
        touch("results/training_data/{filtering_mode}/hrs{window}/.done")

    log:
        "logs/compute_motif_enrichment/{filtering_mode}_hrs{window}.log"

    conda:
        "../../env/motif_enrichment.yaml"

    shell:
        """
        PYTHONPATH=src/py python src/py/00_prepare_data/compute_motif_enrichment.py \
            --window {wildcards.window} \
            --filtering_mode {wildcards.filtering_mode} \
            --metadata_path {input.metadata} \
            --output_dir results/training_data/{wildcards.filtering_mode}/hrs{wildcards.window} \
            --log_path {log}
        """
