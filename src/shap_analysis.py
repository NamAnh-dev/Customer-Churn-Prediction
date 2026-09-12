"""SHAP-based model explainability + business ROI estimation.

Key fix vs the original file: every plotting function used to assume its
own shape for `shap_values` (some expected 2D, some expected 3D
`(n_samples, n_features, n_classes)`), and those assumptions disagreed with
each other. SHAP's own API has changed this shape across versions, so the
two assumptions silently breaking was a matter of *when*, not *if*.
`_flatten_shap_output()` is now the single place that normalizes whatever
SHAP returns into one consistent 2D array (rows = samples, cols = features,
values = contribution to the POSITIVE / churn class). Every function below
calls it, so there is exactly one place to fix if a future SHAP version
changes its output format again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)


def _flatten_shap_output(shap_vals, positive_class: int = 1) -> np.ndarray:
    """Normalize any SHAP output shape into a 2D (n_samples, n_features) array
    of contributions toward `positive_class`.

    Handles the three shapes SHAP has used across versions:
    - list of arrays, one per class: [array(n, f), array(n, f)]
    - single 3D array: (n_samples, n_features, n_classes)
    - already 2D: (n_samples, n_features)  (e.g. KernelExplainer on predict_proba
      for a single class, or newer SHAP already returning the positive class only)
    """
    if isinstance(shap_vals, list):
        return np.asarray(shap_vals[positive_class])
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3:
        return shap_vals[:, :, positive_class]
    return shap_vals


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 500):
    """Compute SHAP values for a fitted, already-encoded numeric feature matrix.

    `model` must be the raw estimator (e.g. `pipeline.named_steps["model"]`),
    not the full pipeline — SHAP explainers need direct access to the model's
    `predict`/`predict_proba` and, for tree models, its internal structure.
    `X` must already be through preprocessing (i.e. the pipeline's `preprocess`
    step output), matching exactly what the model was trained on.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is not installed. Run: pip install shap") from e

    X_sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X.copy()
    log.info("Computing SHAP values for %d samples...", len(X_sample))

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_sample)
    else:
        background = shap.kmeans(X_sample, 10)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        raw_shap = explainer.shap_values(X_sample)

    shap_values = _flatten_shap_output(raw_shap)
    log.info("SHAP values computed: shape=%s", shap_values.shape)
    return shap_values, explainer, X_sample


def plot_shap_summary(shap_values: np.ndarray, X_sample: pd.DataFrame, save_path: str = None, max_display: int = 15):
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is not installed. Run: pip install shap") from e

    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False, plot_size=None)
    plt.title("SHAP Summary Plot — Feature Impact on Churn Probability", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_importance(shap_values: np.ndarray, X_sample: pd.DataFrame, save_path: str = None, top_n: int = 15):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # 2D input guaranteed -> average over samples only
    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "importance": mean_abs_shap,
    }).sort_values("importance").tail(top_n)

    def get_color(feat: str) -> str:
        account_feats = ["Contract", "tenure", "MonthlyCharges", "TotalCharges",
                          "PaperlessBilling", "PaymentMethod", "charge_ratio",
                          "avg_monthly_charge", "num_services"]
        service_feats = ["InternetService", "OnlineSecurity", "TechSupport",
                          "StreamingTV", "StreamingMovies", "OnlineBackup",
                          "DeviceProtection", "MultipleLines", "PhoneService"]
        if any(a.lower() in feat.lower() for a in account_feats):
            return "#378ADD"
        if any(s.lower() in feat.lower() for s in service_feats):
            return "#1D9E75"
        return "#7F77DD"

    colors = [get_color(f) for f in importance_df["feature"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance_df["feature"], importance_df["importance"], color=colors, height=0.65)
    ax.set_title("SHAP Feature Importance\nmean(|SHAP value|) across sampled customers",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.spines[["top", "right"]].set_visible(False)

    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#378ADD", label="Account features"),
        Patch(color="#1D9E75", label="Service features"),
        Patch(color="#7F77DD", label="Demographics"),
    ], fontsize=9, loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_dependence(shap_values: np.ndarray, X_sample: pd.DataFrame, feature: str,
                          interaction_feature: str = None, save_path: str = None):
    if feature not in X_sample.columns:
        log.warning("Feature '%s' not found in X_sample", feature)
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    feature_idx = list(X_sample.columns).index(feature)

    x_vals = X_sample[feature].values
    y_vals = shap_values[:, feature_idx]  # 2D input guaranteed, no extra axis to average over

    if interaction_feature and interaction_feature in X_sample.columns:
        interact_vals = X_sample[interaction_feature].values
        scatter = ax.scatter(x_vals, y_vals, c=interact_vals, cmap="RdYlGn_r", alpha=0.5, s=15)
        plt.colorbar(scatter, ax=ax, label=interaction_feature)
    else:
        ax.scatter(x_vals, y_vals, color="#378ADD", alpha=0.4, s=15)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value ({feature})")
    ax.set_title(f"SHAP Dependence: {feature}\nPositive = increases churn probability, negative = decreases it",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_waterfall_single(explainer, X_sample: pd.DataFrame, customer_idx: int = 0, save_path: str = None):
    """Explain a single customer's prediction as a waterfall of feature contributions."""
    customer = X_sample.iloc[[customer_idx]]

    raw_shap_single = explainer.shap_values(customer)
    shap_single = _flatten_shap_output(raw_shap_single)[0]  # -> 1D array, one value per feature

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value)[1] if len(np.asarray(expected_value)) > 1 else float(np.asarray(expected_value).mean())
    expected_value = float(expected_value)

    feature_names = list(X_sample.columns)
    feature_values = customer.values[0]

    top_idx = np.argsort(np.abs(shap_single))[-10:][::-1]
    top_shap = shap_single[top_idx]
    top_names = [f"{feature_names[i]}={feature_values[i]:.2g}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#D85A30" if v > 0 else "#1D9E75" for v in top_shap]
    y_pos = range(len(top_shap))

    ax.barh(y_pos, top_shap[::-1], color=colors[::-1], height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Value (impact on churn probability)")

    predicted_prob = expected_value + shap_single.sum()
    ax.set_title(
        f"SHAP Waterfall — Customer #{customer_idx}\n"
        f"Baseline: {expected_value:.2f} | Predicted churn prob: {predicted_prob:.2f}",
        fontsize=11, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compute_roi_table(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    avg_monthly_revenue: float = 65.0,
    avg_tenure_lost: float = 12.0,
    retention_cost: float = 50.0,
    retention_success_rate: float = 0.30,
) -> pd.DataFrame:
    """Translate confusion-matrix outcomes into an estimated dollar impact.

    All dollar assumptions (avg_monthly_revenue, retention_cost,
    retention_success_rate, avg_tenure_lost) are illustrative placeholders,
    not values derived from this dataset (the Telco data has no retention
    campaign cost/success history to estimate them from). State that
    explicitly wherever this table is presented — the *mechanism* (how a
    threshold choice translates to dollars) is the deliverable, not these
    specific numbers. Swap in real figures from Finance/Marketing before
    using this to argue for a specific threshold in production.

    IMPORTANT: pass the threshold you actually intend to deploy (the one
    tuned on the validation set), not the default 0.5, so this table tells
    a consistent story with the rest of the analysis.
    """
    from sklearn.metrics import confusion_matrix

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    revenue_saved_per_customer = avg_monthly_revenue * avg_tenure_lost
    revenue_per_tp = revenue_saved_per_customer * retention_success_rate

    model_cost = (tp + fp) * retention_cost
    model_saved = tp * revenue_per_tp
    model_net = model_saved - model_cost
    no_model_loss = (tp + fn) * revenue_saved_per_customer

    rows = [
        ("Correctly caught churners (TP)", tp, f"+${tp * revenue_per_tp:,.0f}"),
        ("False alarms (FP) — retention cost spent", fp, f"-${fp * retention_cost:,.0f}"),
        ("Missed churners (FN) — revenue lost", fn, f"-${fn * revenue_saved_per_customer:,.0f}"),
        ("Total retention spend", tp + fp, f"-${model_cost:,.0f}"),
        ("Total revenue saved", "-", f"+${model_saved:,.0f}"),
        ("Net benefit vs. no model", "-", f"${model_net:,.0f}"),
        ("Revenue lost with no model at all", tp + fn, f"-${no_model_loss:,.0f}"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Count", "Amount"])
