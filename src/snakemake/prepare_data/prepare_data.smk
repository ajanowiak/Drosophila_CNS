# Stage 0 - prepare_data: computes per-window motif enrichment matrices and
# extracts binary loop-presence vectors. Every later stage builds on these.

rule all_prepare_data:
    input:
        prepare_data_targets()


include: "compute_motif_enrichment.smk"
include: "extract_loop_presence_vectors.smk"
