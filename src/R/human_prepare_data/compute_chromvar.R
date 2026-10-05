#!/usr/bin/env Rscript

# Compute a motif x cell chromVAR deviation matrix (Schep et al. 2017) for the
# selected nuclei. Reads the SnapATAC .snap peak matrix, keeps the chosen cells and
# accessible main-chromosome peaks, scores JASPAR CORE vertebrate motifs, and writes
# the deviation Z-scores as a motif x cell TSV.

suppressPackageStartupMessages({
  library(optparse)
  library(Matrix)
  library(rhdf5)
  library(GenomicRanges)
  library(SummarizedExperiment)
  library(Rsamtools)
  library(TFBSTools)
  library(motifmatchr)
  library(chromVAR)
  library(BiocParallel)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--snap", type = "character",
              help = "SnapATAC .snap HDF5 (GSE244618 processed data)."),
  make_option("--metadata", type = "character",
              default = "data/human_prototype/interim/cell_metadata.tsv",
              help = "Per-nucleus QC + atlas labels."),
  make_option("--cell_label", type = "character", default = "all",
              help = "Value to keep in --label_column, or 'all' for every QC nucleus."),
  make_option("--label_column", type = "character", default = "label",
              help = "Metadata column matched against --cell_label."),
  make_option("--genome", type = "character",
              default = "data/human_prototype/ref/hg38.fa",
              help = "hg38 FASTA; a .fai index must sit next to it."),
  make_option("--motifs", type = "character",
              help = "JASPAR PFM flat file (TFBSTools::readJASPARMatrix format)."),
  make_option("--min_cells_per_peak", type = "integer", default = 1L,
              help = "Keep peaks accessible in at least this many cells (1 = chromVAR default)."),
  make_option("--bg_iterations", type = "integer", default = 50L,
              help = "Number of background peak sets (chromVAR default)."),
  make_option("--seed", type = "integer", default = 0L,
              help = "Seed for background peak sampling."),
  make_option("--threads", type = "integer", default = 1L,
              help = "BiocParallel workers; >1 raises memory use."),
  make_option("--output", type = "character",
              default = "data/human_prototype/interim/motifs_chromvar_all.tsv"),
  make_option("--log_path", type = "character",
              default = "logs/human_prototype/compute_chromvar.log")
)))

dir.create(dirname(opt$log_path), recursive = TRUE, showWarnings = FALSE)
log_con <- file(opt$log_path, open = "at")
logmsg <- function(...) {
  line <- sprintf("%s compute_chromvar.R INFO %s",
                  format(Sys.time(), "%Y-%m-%d %H:%M:%S"), paste0(...))
  cat(line, "\n", sep = "")
  writeLines(line, log_con)
  flush(log_con)
}

read_snap_peak_matrix <- function(snap) {
  barcodes <- h5read(snap, "/BD/name")
  chrom <- h5read(snap, "/PM/peakChrom")
  start <- as.integer(h5read(snap, "/PM/peakStart"))
  end <- as.integer(h5read(snap, "/PM/peakEnd"))
  # /PM stores the cell x peak matrix as 1-based (idx, idy, count) triplets.
  idx <- as.integer(h5read(snap, "/PM/idx"))
  idy <- as.integer(h5read(snap, "/PM/idy"))
  val <- as.numeric(h5read(snap, "/PM/count"))
  counts <- sparseMatrix(i = idx, j = idy, x = val,
                         dims = c(length(barcodes), length(chrom)))
  # peakStart is 0-based; GRanges is 1-based inclusive.
  peaks <- GRanges(chrom, IRanges(start = start + 1L, end = end))
  list(counts = counts, peaks = peaks, barcodes = barcodes)
}

MAIN_CHROMS <- c(paste0("chr", 1:22), "chrX")

if (is.null(opt$snap)) stop("--snap is required")
if (is.null(opt$motifs)) stop("--motifs is required")
snap <- read_snap_peak_matrix(opt$snap)
logmsg(sprintf("Read .snap: %d cells x %d peaks",
               length(snap$barcodes), length(snap$peaks)))

meta <- read.table(opt$metadata, header = TRUE, sep = "\t",
                   stringsAsFactors = FALSE, quote = "")
meta$pass_qc <- as.logical(meta$pass_qc)
keep <- meta$pass_qc
if (opt$cell_label != "all") keep <- keep & meta[[opt$label_column]] == opt$cell_label
keep_barcodes <- meta$barcode[keep]
logmsg(sprintf("Target %s=%s cells: %d", opt$label_column, opt$cell_label, length(keep_barcodes)))

cell_mask <- snap$barcodes %in% keep_barcodes
counts <- snap$counts[cell_mask, , drop = FALSE]
barcodes <- snap$barcodes[cell_mask]

# Keep peaks on the main chromosomes and accessible in at least min_cells_per_peak
# cells (default 1 keeps every accessible peak).
cells_per_peak <- Matrix::colSums(counts > 0)
peak_mask <- as.character(seqnames(snap$peaks)) %in% MAIN_CHROMS &
             cells_per_peak >= opt$min_cells_per_peak
counts <- counts[, peak_mask, drop = FALSE]
peaks <- snap$peaks[peak_mask]
logmsg(sprintf("Kept %d cells x %d peaks (accessible in >= %d cells)",
               nrow(counts), ncol(counts), opt$min_cells_per_peak))

# chromVAR's SummarizedExperiment is peak x cell, with a "counts" assay.
rse <- SummarizedExperiment(assays = list(counts = t(counts)),
                            rowRanges = peaks,
                            colData = DataFrame(barcode = barcodes,
                                                row.names = barcodes))
rse <- sort(rse)

# An indexed FASTA lets chromVAR read sequence without the BSgenome.Hsapiens
# data package.
genome <- FaFile(opt$genome)

logmsg("addGCBias ...")
rse <- addGCBias(rse, genome = genome)
# Peaks over assembly gaps have all-N sequence and undefined GC; drop them so they
# cannot corrupt the background sampling.
ok <- !is.na(rowData(rse)$bias)
if (any(!ok)) logmsg(sprintf("Dropping %d peaks with undefined GC bias", sum(!ok)))
rse <- rse[ok, ]

# Standard chromVAR peak filter: drop empty and overlapping peaks.
rse <- filterPeaks(rse, non_overlapping = TRUE)
logmsg(sprintf("After filterPeaks: %d peaks", nrow(rse)))

logmsg(sprintf("getBackgroundPeaks (niterations=%d, seed=%d) ...", opt$bg_iterations, opt$seed))
set.seed(opt$seed)
bg <- getBackgroundPeaks(rse, niterations = opt$bg_iterations)

motifs <- readJASPARMatrix(opt$motifs, matrixClass = "PFM")
# Label each motif "matrix_id.name" (e.g. MA0004.1.Arnt) for stable deviation rows.
names(motifs) <- vapply(motifs, function(m) paste0(ID(m), ".", name(m)), character(1))
logmsg(sprintf("matchMotifs: %d motifs ...", length(motifs)))
matches <- matchMotifs(motifs, rse, genome = genome)

register(if (opt$threads > 1L) MulticoreParam(workers = opt$threads) else SerialParam())

logmsg(sprintf("computeDeviations (threads=%d) ...", opt$threads))
dev <- computeDeviations(object = rse, annotations = matches, background_peaks = bg)

z <- deviationScores(dev)
colnames(z) <- colnames(rse)
logmsg(sprintf("chromVAR deviations: %d motifs x %d cells", nrow(z), ncol(z)))

dir.create(dirname(opt$output), recursive = TRUE, showWarnings = FALSE)
# col.names = NA prepends an empty header cell so the first column holds motif ids.
write.table(z, file = opt$output, sep = "\t", quote = FALSE,
            col.names = NA, row.names = TRUE)
logmsg(sprintf("Wrote %s", opt$output))
close(log_con)
