from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "plotting.ipynb"


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook():
    cells = [
        markdown_cell(
            """
            # PyTorch Model Plotting

            This notebook loads the PyTorch checkpoints created from the converted training notebooks and rebuilds
            the main comparison plots for all six model variants:

            - Bias initialization
            - Greedy supervised pretraining
            - Greedy unsupervised pretraining
            - SSL shallow autoencoder
            - SSL deep autoencoder
            - SSL contrastive encoder

            It also saves richer per-model figures to `plot_outputs/`, including split-wise probability panels,
            individual ROC curves, individual confusion matrices, and text architecture summaries.
            """
        ),
        code_cell(
            """
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import torch
            import torch.nn.functional as F
            from matplotlib.ticker import PercentFormatter
            from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

            from utils import *

            sns.set_theme(style="whitegrid")
            plt.rcParams["figure.dpi"] = 150
            plt.rcParams["savefig.dpi"] = 150

            print(f"Using device: {device}")
            """
        ),
        code_cell(
            """
            set_seed(0)

            DATA_DIR = Path("./classification_data")
            PLOT_DIR = Path("./plot_outputs")
            PLOT_DIR.mkdir(exist_ok=True)

            required_files = [
                DATA_DIR / "X_train.npy",
                DATA_DIR / "X_val.npy",
                DATA_DIR / "X_test.npy",
                DATA_DIR / "y_train.npy",
                DATA_DIR / "y_val.npy",
                DATA_DIR / "y_test.npy",
                DATA_DIR / "bcu_samples.npy",
            ]
            missing_files = [str(path) for path in required_files if not path.exists()]
            if missing_files:
                raise FileNotFoundError(f"Missing preprocessed files: {missing_files}")

            X_train = np.load(DATA_DIR / "X_train.npy")
            X_val = np.load(DATA_DIR / "X_val.npy")
            X_test = np.load(DATA_DIR / "X_test.npy")
            y_train = np.load(DATA_DIR / "y_train.npy")
            y_val = np.load(DATA_DIR / "y_val.npy")
            y_test = np.load(DATA_DIR / "y_test.npy")
            bcu_samples = np.load(DATA_DIR / "bcu_samples.npy")

            combined_X = np.vstack((X_train, X_val, X_test, bcu_samples))

            split_features = {
                "train": X_train,
                "val": X_val,
                "test": X_test,
                "bcu": bcu_samples,
                "combined": combined_X,
            }
            split_labels = {
                "train": y_train,
                "val": y_val,
                "test": y_test,
                "bcu": None,
                "combined": None,
            }
            split_titles = {
                "train": "Train",
                "val": "Validation",
                "test": "Test",
                "bcu": "BCU",
                "combined": "Combined",
            }

            bll, fsrq = np.bincount(y_train.astype(int))
            output_bias = float(np.log(fsrq / bll))
            print({"bll": int(bll), "fsrq": int(fsrq), "output_bias": output_bias})
            """
        ),
        code_cell(
            """
            class EncoderWrapper(nn.Module):
                def __init__(self, autoencoder):
                    super().__init__()
                    self.autoencoder = autoencoder

                def forward(self, x):
                    return self.autoencoder.encode(x)


            class ContrastiveClassifier(nn.Module):
                def __init__(self, input_dim, shared_dim=64):
                    super().__init__()
                    self.shared_dense = nn.Linear(input_dim, shared_dim)
                    self.output_layer = nn.Linear(shared_dim * 2, 1)

                def encode(self, x):
                    return F.relu(self.shared_dense(x))

                def forward(self, x_a, x_b):
                    encoded_a = self.encode(x_a)
                    encoded_b = self.encode(x_b)
                    merged = torch.cat([encoded_a, encoded_b], dim=1)
                    return self.output_layer(merged)


            class ContrastiveEncoder(nn.Module):
                def __init__(self, contrastive_model):
                    super().__init__()
                    self.contrastive_model = contrastive_model

                def forward(self, x):
                    return self.contrastive_model.encode(x)


            def count_parameters(model):
                return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


            def architecture_summary(name, model):
                lines = [
                    name,
                    "-" * len(name),
                    str(model),
                    f"Trainable parameters: {count_parameters(model):,}",
                ]
                return "\\n".join(lines)


            def bias_factory():
                return MLP(
                    input_dim=X_train.shape[1],
                    hidden_dims=[42],
                    output_dim=1,
                    dropout_rates=[0.5],
                    output_bias=output_bias,
                )


            def greedy_factory():
                return MLP(
                    input_dim=X_train.shape[1],
                    hidden_dims=[64, 32],
                    output_dim=1,
                    dropout_rates=[0.5, 0.0],
                )


            def shallow_autoencoder_factory():
                return Autoencoder(
                    input_dim=X_train.shape[1],
                    encoder_dims=[64],
                    latent_dim=32,
                    decoder_dims=[64],
                    encoder_dropout=[0.0],
                )


            def deep_autoencoder_factory():
                return Autoencoder(
                    input_dim=X_train.shape[1],
                    encoder_dims=[128, 64],
                    latent_dim=32,
                    decoder_dims=[64, 128],
                    encoder_dropout=[0.0, 0.0],
                )


            def contrastive_pair_factory():
                return ContrastiveClassifier(input_dim=X_train.shape[1], shared_dim=64)


            def shallow_encoder_model():
                autoencoder = shallow_autoencoder_factory()
                load_checkpoint(autoencoder, resolve_checkpoint_path("./ssl_autoencoder_ckpts_5epochs", 5))
                return EncoderWrapper(autoencoder)


            def deep_encoder_model():
                autoencoder = deep_autoencoder_factory()
                load_checkpoint(autoencoder, resolve_checkpoint_path("./ssl_autoencoder_deep_ckpts_5epochs", 5))
                return EncoderWrapper(autoencoder)


            def contrastive_encoder_model():
                contrastive_model = contrastive_pair_factory()
                load_checkpoint(contrastive_model, resolve_checkpoint_path("./ssl_contrastive_pair_ckpts_5epochs", 5))
                return ContrastiveEncoder(contrastive_model)


            def classifier_head_factory(input_dim):
                return MLP(input_dim=input_dim, hidden_dims=[], output_dim=1)


            def read_checkpoint_history(checkpoint_dir, epoch):
                checkpoint_path = resolve_checkpoint_path(checkpoint_dir, epoch)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict):
                    return checkpoint.get("history", {})
                return {}


            def predict_checkpoint_bundle(model_factory, checkpoint_dir, eval_features, eval_labels, feature_map):
                _, best_metrics = evaluate_checkpoint_directory(model_factory, checkpoint_dir, eval_features, eval_labels)
                best_epoch = best_metrics["auc"]["epoch"]
                model = model_factory()
                load_checkpoint(model, resolve_checkpoint_path(checkpoint_dir, best_epoch), map_location=device)
                outputs = {
                    split_name: np.squeeze(predict_array(model, split_array, apply_sigmoid=True))
                    for split_name, split_array in feature_map.items()
                }
                history = read_checkpoint_history(checkpoint_dir, best_epoch)
                return outputs, best_metrics, best_epoch, history


            def encode_feature_map(encoder, feature_map):
                return {
                    split_name: predict_array(encoder, split_array)
                    for split_name, split_array in feature_map.items()
                }


            def load_all_predictions():
                predictions = {}
                metrics_rows = []

                bias_outputs = {
                    split_name: np.squeeze(
                        soft_voting_predictions(
                            bias_factory,
                            "./checkpoints_classifier_379_params_2epochs",
                            [1, 2],
                            [0.1, 0.9],
                            split_array,
                        )
                    )
                    for split_name, split_array in split_features.items()
                }
                bias_metrics = compute_binary_classification_metrics(y_test, bias_outputs["test"])
                predictions["Bias Init"] = {
                    "outputs": bias_outputs,
                    "selection": "soft vote [1, 2]",
                    "history": read_checkpoint_history("./checkpoints_classifier_379_params_2epochs", 2),
                    "metrics": bias_metrics,
                }
                metrics_rows.append(
                    {
                        "model": "Bias Init",
                        "selection": "soft vote [1, 2]",
                        "val_best_auc": np.nan,
                        **bias_metrics,
                    }
                )

                greedy_outputs, greedy_best, greedy_epoch, greedy_history = predict_checkpoint_bundle(
                    greedy_factory,
                    "./greedy_ckpts_5epochs",
                    X_test,
                    y_test,
                    split_features,
                )
                greedy_metrics = compute_binary_classification_metrics(y_test, greedy_outputs["test"])
                predictions["Greedy Sup"] = {
                    "outputs": greedy_outputs,
                    "selection": f"epoch {greedy_epoch}",
                    "history": greedy_history,
                    "metrics": greedy_metrics,
                }
                metrics_rows.append(
                    {
                        "model": "Greedy Sup",
                        "selection": f"epoch {greedy_epoch}",
                        "val_best_auc": greedy_best["auc"]["score"],
                        **greedy_metrics,
                    }
                )

                unsup_outputs, unsup_best, unsup_epoch, unsup_history = predict_checkpoint_bundle(
                    greedy_factory,
                    "./greedy_ckpts_unsup_5epochs",
                    X_test,
                    y_test,
                    split_features,
                )
                unsup_metrics = compute_binary_classification_metrics(y_test, unsup_outputs["test"])
                predictions["Greedy Unsup"] = {
                    "outputs": unsup_outputs,
                    "selection": f"epoch {unsup_epoch}",
                    "history": unsup_history,
                    "metrics": unsup_metrics,
                }
                metrics_rows.append(
                    {
                        "model": "Greedy Unsup",
                        "selection": f"epoch {unsup_epoch}",
                        "val_best_auc": unsup_best["auc"]["score"],
                        **unsup_metrics,
                    }
                )

                shallow_encoder = shallow_encoder_model()
                shallow_feature_map = encode_feature_map(shallow_encoder, split_features)
                shallow_factory = lambda: classifier_head_factory(shallow_feature_map["train"].shape[1])
                shallow_outputs, shallow_best, shallow_epoch, shallow_history = predict_checkpoint_bundle(
                    shallow_factory,
                    "./ssl_ckpt_no_dropout_5epochs",
                    shallow_feature_map["test"],
                    y_test,
                    shallow_feature_map,
                )
                shallow_metrics = compute_binary_classification_metrics(y_test, shallow_outputs["test"])
                predictions["SSL Shallow"] = {
                    "outputs": shallow_outputs,
                    "selection": f"epoch {shallow_epoch}",
                    "history": shallow_history,
                    "metrics": shallow_metrics,
                }
                metrics_rows.append(
                    {
                        "model": "SSL Shallow",
                        "selection": f"epoch {shallow_epoch}",
                        "val_best_auc": shallow_best["auc"]["score"],
                        **shallow_metrics,
                    }
                )

                deep_encoder = deep_encoder_model()
                deep_feature_map = encode_feature_map(deep_encoder, split_features)
                deep_factory = lambda: classifier_head_factory(deep_feature_map["train"].shape[1])
                deep_outputs, deep_best, deep_epoch, deep_history = predict_checkpoint_bundle(
                    deep_factory,
                    "./ssl_ckpt_deep_5epochs",
                    deep_feature_map["test"],
                    y_test,
                    deep_feature_map,
                )
                deep_metrics = compute_binary_classification_metrics(y_test, deep_outputs["test"])
                predictions["SSL Deep"] = {
                    "outputs": deep_outputs,
                    "selection": f"epoch {deep_epoch}",
                    "history": deep_history,
                    "metrics": deep_metrics,
                }
                metrics_rows.append(
                    {
                        "model": "SSL Deep",
                        "selection": f"epoch {deep_epoch}",
                        "val_best_auc": deep_best["auc"]["score"],
                        **deep_metrics,
                    }
                )

                contrastive_encoder = contrastive_encoder_model()
                contrastive_feature_map = encode_feature_map(contrastive_encoder, split_features)
                contrastive_factory = lambda: classifier_head_factory(contrastive_feature_map["train"].shape[1])
                contrastive_outputs, contrastive_best, contrastive_epoch, contrastive_history = predict_checkpoint_bundle(
                    contrastive_factory,
                    "./ssl_contrastive_ckpt_5epochs",
                    contrastive_feature_map["test"],
                    y_test,
                    contrastive_feature_map,
                )
                contrastive_metrics = compute_binary_classification_metrics(y_test, contrastive_outputs["test"])
                predictions["SSL Contrastive"] = {
                    "outputs": contrastive_outputs,
                    "selection": f"epoch {contrastive_epoch}",
                    "history": contrastive_history,
                    "metrics": contrastive_metrics,
                }
                metrics_rows.append(
                    {
                        "model": "SSL Contrastive",
                        "selection": f"epoch {contrastive_epoch}",
                        "val_best_auc": contrastive_best["auc"]["score"],
                        **contrastive_metrics,
                    }
                )

                metrics_df = pd.DataFrame(metrics_rows)
                return predictions, metrics_df
            """
        ),
        code_cell(
            """
            def slugify(name):
                return name.lower().replace(" ", "_")


            def build_confusion_annotations(cm):
                labels = np.empty_like(cm, dtype=object)
                row_totals = cm.sum(axis=1, keepdims=True)
                for row_idx in range(cm.shape[0]):
                    for col_idx in range(cm.shape[1]):
                        total = row_totals[row_idx, 0]
                        fraction = (cm[row_idx, col_idx] / total) if total else 0.0
                        labels[row_idx, col_idx] = f"{cm[row_idx, col_idx]}\\n{fraction:.1%}"
                return labels


            def plot_probability_histogram(ax, probabilities, labels=None, title=None, bins=16):
                probabilities = np.asarray(probabilities).reshape(-1)
                if labels is None:
                    weights = np.ones_like(probabilities, dtype=float) / len(probabilities)
                    ax.hist(
                        probabilities,
                        bins=bins,
                        range=(0.0, 1.0),
                        weights=weights,
                        color="tab:blue",
                        alpha=0.45,
                        edgecolor="white",
                    )
                else:
                    labels = np.asarray(labels).reshape(-1).astype(int)
                    for value, color, label in [
                        (0, "tab:red", "BL Lac"),
                        (1, "tab:blue", "FSRQ"),
                    ]:
                        mask = labels == value
                        if not np.any(mask):
                            continue
                        weights = np.ones(mask.sum(), dtype=float) / mask.sum()
                        ax.hist(
                            probabilities[mask],
                            bins=bins,
                            range=(0.0, 1.0),
                            weights=weights,
                            histtype="step",
                            linewidth=2.0,
                            color=color,
                            label=label,
                        )

                ax.axvline(0.5, linestyle="--", linewidth=1.0, color="black")
                ax.set_xlim(0.0, 1.0)
                ax.set_xticks(np.linspace(0.0, 1.0, 5))
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
                ax.set_xlabel("Predicted FSRQ Probability")
                ax.set_ylabel("Fraction of Sources")
                if title is not None:
                    ax.set_title(title)


            def save_distribution_panel(model_name, model_info):
                slug = slugify(model_name)
                figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
                split_order = ["train", "val", "test", "bcu", "combined"]
                for ax, split_name in zip(axes.flat[:5], split_order):
                    probabilities = model_info["outputs"][split_name]
                    labels = split_labels[split_name]
                    title = f"{split_titles[split_name]} (n={len(probabilities)})"
                    plot_probability_histogram(ax, probabilities, labels=labels, title=title)
                    if split_name == "train":
                        handles, labels_text = ax.get_legend_handles_labels()
                        if handles:
                            ax.legend(frameon=False)

                metrics = model_info["metrics"]
                summary_lines = [
                    model_name,
                    f"Selection: {model_info['selection']}",
                    f"Test accuracy: {metrics['accuracy']:.3f}",
                    f"Test AUC: {metrics['auc']:.3f}",
                    f"Weighted F1: {metrics['f1']:.3f}",
                ]
                axes.flat[5].text(
                    0.03,
                    0.97,
                    "\\n".join(summary_lines),
                    va="top",
                    ha="left",
                    fontsize=12,
                    family="monospace",
                )
                axes.flat[5].axis("off")
                figure.suptitle(f"{model_name}: Split-wise Prediction Distributions", fontsize=18)

                output_path = PLOT_DIR / f"{slug}_split_panels.png"
                figure.savefig(output_path, bbox_inches="tight")
                plt.show()
                return output_path


            def save_roc_plot(model_name, probabilities):
                slug = slugify(model_name)
                auc_value = roc_auc_score(y_test, probabilities)
                fpr, tpr, _ = roc_curve(y_test, probabilities)

                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, linewidth=2.0, label=f"{model_name} (AUC={auc_value:.3f})")
                plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"{model_name}: ROC Curve")
                plt.legend(frameon=False)
                plt.tight_layout()

                output_path = PLOT_DIR / f"{slug}_roc.png"
                plt.savefig(output_path, bbox_inches="tight")
                plt.show()
                return output_path


            def save_confusion_plot(model_name, probabilities):
                slug = slugify(model_name)
                predicted_labels = (np.asarray(probabilities).reshape(-1) >= 0.5).astype(int)
                cm = confusion_matrix(y_test, predicted_labels, labels=[0, 1])
                annotations = build_confusion_annotations(cm)

                plt.figure(figsize=(6, 5))
                ax = sns.heatmap(
                    cm,
                    annot=annotations,
                    fmt="",
                    cmap="Blues",
                    cbar=False,
                    square=True,
                    linewidths=0.75,
                    linecolor="white",
                    xticklabels=["BLL", "FSRQ"],
                    yticklabels=["BLL", "FSRQ"],
                )
                ax.set_xlabel("Predicted Class")
                ax.set_ylabel("True Class")
                ax.set_title(f"{model_name}: Confusion Matrix\\n(count and row fraction)")
                plt.tight_layout()

                output_path = PLOT_DIR / f"{slug}_confusion.png"
                plt.savefig(output_path, bbox_inches="tight")
                plt.show()
                return output_path


            def save_architecture_summaries():
                model_summaries = [
                    ("Bias Init", bias_factory()),
                    ("Greedy Classifier", greedy_factory()),
                    ("SSL Shallow Autoencoder", shallow_autoencoder_factory()),
                    ("SSL Deep Autoencoder", deep_autoencoder_factory()),
                    ("SSL Contrastive Pair Model", contrastive_pair_factory()),
                    ("SSL Linear Head (32-d)", classifier_head_factory(32)),
                    ("SSL Linear Head (64-d)", classifier_head_factory(64)),
                ]
                summary_text = "\\n\\n".join(architecture_summary(name, model) for name, model in model_summaries)
                output_path = PLOT_DIR / "architecture_summaries.txt"
                output_path.write_text(summary_text)
                print(summary_text)
                return output_path
            """
        ),
        code_cell(
            """
            architecture_path = save_architecture_summaries()
            architecture_path
            """
        ),
        code_cell(
            """
            predictions, metrics_df = load_all_predictions()
            metrics_df
            """
        ),
        code_cell(
            """
            artifact_rows = []
            for model_name, model_info in predictions.items():
                split_panel_path = save_distribution_panel(model_name, model_info)
                roc_path = save_roc_plot(model_name, model_info["outputs"]["test"])
                confusion_path = save_confusion_plot(model_name, model_info["outputs"]["test"])
                artifact_rows.append(
                    {
                        "model": model_name,
                        "split_panels": str(split_panel_path),
                        "roc": str(roc_path),
                        "confusion": str(confusion_path),
                    }
                )

            artifact_df = pd.DataFrame(artifact_rows)
            artifact_df
            """
        ),
        code_cell(
            """
            def add_model_grid_histograms(split_name, title, labels=None, filename=None):
                figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
                for ax, (model_name, model_info) in zip(axes.flat, predictions.items()):
                    plot_probability_histogram(
                        ax,
                        model_info["outputs"][split_name],
                        labels=labels,
                        title=model_name,
                    )
                figure.suptitle(title, fontsize=18)
                output_path = PLOT_DIR / filename
                figure.savefig(output_path, bbox_inches="tight")
                plt.show()
                return output_path


            test_hist_path = add_model_grid_histograms(
                "test",
                "Testing-set Prediction Histograms",
                labels=y_test,
                filename="test_histograms_grid.png",
            )
            bcu_hist_path = add_model_grid_histograms(
                "bcu",
                "BCU Prediction Histograms",
                labels=None,
                filename="bcu_histograms_grid.png",
            )

            plt.figure(figsize=(8, 6))
            for model_name, model_info in predictions.items():
                test_probabilities = model_info["outputs"]["test"]
                fpr, tpr, _ = roc_curve(y_test, test_probabilities)
                auc_value = roc_auc_score(y_test, test_probabilities)
                plt.plot(fpr, tpr, linewidth=2.0, label=f"{model_name} (AUC={auc_value:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Comparison on Test Set")
            plt.legend(fontsize=9, frameon=False)
            roc_grid_path = PLOT_DIR / "roc_comparison.png"
            plt.savefig(roc_grid_path, bbox_inches="tight")
            plt.show()

            figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
            for ax, (model_name, model_info) in zip(axes.flat, predictions.items()):
                cm = confusion_matrix(y_test, (model_info["outputs"]["test"] >= 0.5).astype(int), labels=[0, 1])
                annotations = build_confusion_annotations(cm)
                sns.heatmap(
                    cm,
                    annot=annotations,
                    fmt="",
                    cmap="Blues",
                    cbar=False,
                    square=True,
                    linewidths=0.75,
                    linecolor="white",
                    xticklabels=["BLL", "FSRQ"],
                    yticklabels=["BLL", "FSRQ"],
                    ax=ax,
                )
                ax.set_title(model_name)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
            figure.suptitle("Confusion Matrices on Test Set", fontsize=18)
            confusion_grid_path = PLOT_DIR / "confusion_matrices_grid.png"
            figure.savefig(confusion_grid_path, bbox_inches="tight")
            plt.show()

            pd.DataFrame(metrics_df).to_csv(PLOT_DIR / "model_metrics.csv", index=False)
            artifact_df.to_csv(PLOT_DIR / "plot_manifest.csv", index=False)

            {
                "test_histograms_grid": str(test_hist_path),
                "bcu_histograms_grid": str(bcu_hist_path),
                "roc_comparison": str(roc_grid_path),
                "confusion_matrices_grid": str(confusion_grid_path),
            }
            """
        ),
        code_cell(
            """
            sorted(Path("./plot_outputs").glob("*"))
            """
        ),
    ]

    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    }
    return notebook


def validate_notebook(notebook):
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        ast.parse(cell.source, filename=f"cell_{index}")


def main():
    notebook = build_notebook()
    validate_notebook(notebook)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
