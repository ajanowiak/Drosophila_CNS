import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from utils import print_timestamp


SUPPORTED_MODELS = ["rf", "lr", "xgb", "svc"]
TISSUES = ["Neuroblasts", "Neurons", "Glia"]
WINDOWS = ["06-08", "10-12", "14-16"]


# =========================
# EPV
# =========================
def num_features_from_epv(tissue: str, epv: int) -> int:
    training_data_dir = "results/training_data/unfiltered/"
    timepoints = ["6-8h", "10-12h", "14-16h"]

    y_vector = []

    for w, tp in zip(WINDOWS, timepoints):
        path = os.path.join(training_data_dir, f"hrs{w}/y_{tissue}.csv")
        df = pd.read_csv(path)
        y_vector.extend(df[f"Dmel_{tp}_{tissue}"].to_list())

    num_events = sum(y_vector)
    return int(np.ceil(num_events / epv))


# =========================
# SHAP FEATURE SELECTION
# =========================
def downsample_features_shap(tissue: str, model: str, num_features=None) -> pd.Series:
    shap_path = f"results/shap/{model}/{tissue}_shap_table_{model.upper()}.csv"

    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"Missing SHAP file: {shap_path}")

    shap_df = pd.read_csv(shap_path)

    sorted_features = shap_df.sort_values(
        "abs_mean_importance", ascending=False
    )["motif_id"]

    return sorted_features[:num_features] if num_features else sorted_features


# =========================
# DATA PREP
# =========================
def compose_downsampled_windows(tissue: str, features: pd.Series):
    Xs, ys = [], []
    training_dir = "results/training_data/unfiltered"

    for idx, w in enumerate(WINDOWS):
        X = pd.read_csv(
            os.path.join(training_dir, f"hrs{w}/motif_enrichment_hrs{w}.csv"),
            index_col=0,
        )
        y = pd.read_csv(
            os.path.join(training_dir, f"hrs{w}/y_{tissue}.csv"),
            index_col=0,
        ).iloc[:, 0]

        X = X.dropna(axis=0)
        X = X[features]
        X["window"] = idx

        common = X.index.intersection(y.index)
        X, y = X.loc[common], y.loc[common]

        Xs.append(X)
        ys.append(y)

    X_all = pd.concat(Xs)
    y_all = pd.concat(ys)

    composite = pd.Categorical(list(zip(X_all["window"], y_all))).codes
    X_all.drop("window", axis=1, inplace=True)

    return X_all, y_all, composite


# =========================
# PLOTTING
# =========================
def plot_coeffs(df, tissue, output_base, top_n=20):
    df_plot = df.head(top_n).iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        df_plot["coef"],
        df_plot.index,
        xerr=[
            df_plot["coef"] - df_plot["ci_lower"],
            df_plot["ci_upper"] - df_plot["coef"],
        ],
        fmt="o",
    )

    plt.axvline(0, linestyle="--")
    plt.title(f"{tissue} - Top {top_n} coefficients")
    plt.tight_layout()

    plt.savefig(output_base + ".pdf", dpi=300)
    plt.savefig(output_base + ".png", dpi=300)
    plt.close()


def plot_volcano(df, tissue, output_base):
    p = df["p_adjusted_bh"]

    plt.figure(figsize=(7, 6))
    plt.scatter(df["coef"], -np.log10(p), alpha=0.6)

    plt.axhline(-np.log10(0.05), linestyle="dotted")
    plt.axvline(0, linestyle="dotted")

    plt.xlabel("coef")
    plt.ylabel("-log10(p)")
    plt.title(f"{tissue} volcano")

    plt.tight_layout()

    plt.savefig(output_base + ".pdf", dpi=300)
    plt.savefig(output_base + ".png", dpi=300)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--epv", type=int, default=15)
    args = parser.parse_args()

    model = args.model
    epv = args.epv

    print_timestamp(f"=== regression_coefs | model={model} | epv={epv} ===")

    annot = pd.read_csv("data/motif_names.tsv", sep="\t")
    code_to_name = (
        annot.drop_duplicates("id").set_index("id")["name"].to_dict()
    )
    code_to_name = {k: f"{k}  -  ({v})" for k, v in code_to_name.items()}

    for tissue in TISSUES:
        print_timestamp(f"Processing {tissue}")

        n_features = num_features_from_epv(tissue, epv)
        features = downsample_features_shap(tissue, model, n_features)

        X, y, _ = compose_downsampled_windows(tissue, features)

        X_named = X.copy()
        X_named.columns = [
            code_to_name.get(c, c) for c in X.columns
        ]

        res = sm.Logit(y, sm.add_constant(X_named)).fit(disp=False)

        pvals = res.pvalues
        _, p_bh, _, _ = multipletests(pvals, method="fdr_bh")
        _, p_tsbh, _, _ = multipletests(pvals, method="fdr_tsbh")

        df = pd.DataFrame({
            "coef": res.params,
            "std_err": res.bse,
            "z": res.tvalues,
            "p_unadjusted": pvals,
            "p_adjusted_bh": p_bh,
            "p_adjusted_tsbh": p_tsbh,
        })

        ci = res.conf_int()
        ci.columns = ["ci_lower", "ci_upper"]
        df = pd.concat([df, ci], axis=1)

        df = df.sort_values("coef", key=abs, ascending=False)

        # paths
        data_dir = f"results/regression_coefs/{model}/epv_{epv}"
        fig_dir = f"results/figures/regression_coefs/{model}/epv_{epv}"

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        # save table
        df.to_csv(os.path.join(data_dir, f"{tissue}_summary.csv"))

        # plots
        plot_coeffs(df, tissue,
                    os.path.join(fig_dir, f"{tissue}_coeffs"))

        plot_volcano(df, tissue,
                     os.path.join(fig_dir, f"{tissue}_volcano"))

        print_timestamp(f"{tissue} done")


if __name__ == "__main__":
    main()