import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
)
import logging

log = logging.getLogger(__name__)


def compute_all_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy":  (y_true == y_pred).mean(),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_true, y_prob),
        "PR-AUC":    average_precision_score(y_true, y_prob),
    }


def print_metrics(metrics: dict, model_name: str = ""):
    print(f"\n{'='*50}")
    if model_name:
        print(f"  {model_name}")
        print(f"{'='*50}")
    for k, v in metrics.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<12}: {v:.3f}  {bar}")
    print(f"{'='*50}")


def plot_confusion_matrix(y_true, y_pred,title: str = "Confusion Matrix",save_path: str = None,):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNo Churn", "Predicted\nChurn"], fontsize=11)
    ax.set_yticklabels(["Actual\nNo Churn", "Actual\nChurn"], fontsize=11)

    labels = [
        [f"TN\n{tn:,}\n(Correct)", f"FP\n{fp:,}\n(False alarm)"],
        [f"FN\n{fn:,}\n(MISSED!)",  f"TP\n{tp:,}\n(Caught)"],
    ]
    colors = [["white", "white"], ["#D85A30", "white"]]

    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j],
                    ha="center", va="center", fontsize=11,
                    color=colors[i][j], fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    print(f"\n  Recall    = TP/(TP+FN) = {tp}/{tp+fn} = {recall:.1%}")
    print(f"  Precision = TP/(TP+FP) = {tp}/{tp+fp} = {precision:.1%}")
    print(f"  Missed churners (FN): {fn:,}")


def plot_roc_pr_curves(models_results: dict,y_true,save_path: str = None,):

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#EF9F27"]

    for i, (name, y_prob) in enumerate(models_results.items()):
        color = colors[i % len(colors)]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, color=color, linewidth=2,
                     label=f"{name} (AUC={auc:.3f})")

        # PR Curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        axes[1].plot(rec, prec, color=color, linewidth=2,
                     label=f"{name} (PR-AUC={pr_auc:.3f})")

    # ROC — diagonal baseline
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].set_title("ROC Curve", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # PR — baseline = churn rate
    baseline = y_true.mean()
    axes[1].axhline(baseline, color="k", linestyle="--", linewidth=1, alpha=0.5,
                    label=f"Random ({baseline:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Model Evaluation Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_metrics_comparison(all_metrics: dict,save_path: str = None,):
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_to_plot = ["Precision", "Recall", "F1", "AUC-ROC", "PR-AUC"]
    metrics_df = metrics_df[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics_df))
    width = 0.15
    colors = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#EF9F27"]

    for i, metric in enumerate(metrics_to_plot):
        bars = ax.bar(x + i * width, metrics_df[metric], width,
                      label=metric, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics_df.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Classification Metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("\n=== Metrics Summary ===")
    print(metrics_df.round(3).to_string())


def find_optimal_threshold(y_true, y_prob, metric: str = "f1") -> float:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score, best_thresh = -1, 0.5

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            score = (recall_score(y_true, y_pred, zero_division=0) +
                     precision_score(y_true, y_pred, zero_division=0)) / 2

        if score > best_score:
            best_score = score
            best_thresh = thresh

    log.info(f"Optimal threshold ({metric}): {best_thresh:.2f} → score={best_score:.3f}")
    return best_thresh


def plot_threshold_analysis(y_true, y_prob,save_path: str = None):
    thresholds = np.linspace(0.1, 0.9, 81)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    best_f1_idx = np.argmax(f1s)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, precisions, color="#378ADD", linewidth=2, label="Precision")
    ax.plot(thresholds, recalls,    color="#D85A30", linewidth=2, label="Recall")
    ax.plot(thresholds, f1s,        color="#1D9E75", linewidth=2.5, label="F1-Score")
    ax.axvline(thresholds[best_f1_idx], color="#7F77DD", linestyle="--", linewidth=2,
               label=f"Best F1 threshold = {thresholds[best_f1_idx]:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Default (0.5)")

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return thresholds[best_f1_idx]
