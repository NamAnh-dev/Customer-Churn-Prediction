import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 500):
    try:
        import shap
    except ImportError:
        raise ImportError("Chưa cài shap. Chạy: pip install shap")

    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X.copy()

    log.info(f"Computing SHAP values for {len(X_sample)} samples...")

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    else:
        background = shap.kmeans(X_sample, 10)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    log.info(f"SHAP values computed: shape={shap_vals.shape}")
    return shap_vals, explainer, X_sample


# ── Plot 1: Summary plot ──────────────────────────────────────
def plot_shap_summary(shap_values: np.ndarray,X_sample: pd.DataFrame,save_path: str = None,max_display: int = 15):
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    shap.summary_plot(
        shap_values, X_sample,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary Plot — Feature Impact on Churn Probability",fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Saved SHAP summary plot")


# ── Plot 2: Bar plot (mean |SHAP|) ───────────────────────────────────────
def plot_shap_importance(shap_values: np.ndarray,X_sample: pd.DataFrame,save_path: str = None,top_n: int = 15):
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    # mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    X_sample.columns,
        "importance": mean_abs_shap,
    }).sort_values("importance").tail(top_n)

    def get_color(feat):
        account_feats = ["Contract", "tenure", "MonthlyCharges", "TotalCharges",
                         "PaperlessBilling", "PaymentMethod", "charge_ratio",
                         "avg_monthly_charge", "num_services"]
        service_feats = ["InternetService", "OnlineSecurity", "TechSupport",
                         "StreamingTV", "StreamingMovies", "OnlineBackup",
                         "DeviceProtection", "MultipleLines", "PhoneService"]
        for a in account_feats:
            if a.lower() in feat.lower():
                return "#378ADD"
        for s in service_feats:
            if s.lower() in feat.lower():
                return "#1D9E75"
        return "#7F77DD"

    colors = [get_color(f) for f in importance_df["feature"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance_df["feature"], importance_df["importance"],
                   color=colors, height=0.65)
    ax.set_title("SHAP Feature Importance\nmean(|SHAP value|) — contribution trung bình",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.spines[["top", "right"]].set_visible(False)

    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
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


# ── Plot 3: Dependence plot ───────────────────────────────────────────────
def plot_shap_dependence(shap_values: np.ndarray,X_sample: pd.DataFrame,feature: str,interaction_feature: str = None,save_path: str = None,):
    
    if feature not in X_sample.columns:
        log.warning(f"Feature '{feature}' not found in X_sample")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    feature_idx = list(X_sample.columns).index(feature)

    x_vals = X_sample[feature].values
    y_vals = shap_values[:, feature_idx, :].mean(axis=1)

    if interaction_feature and interaction_feature in X_sample.columns:
        interact_vals = X_sample[interaction_feature].values
        scatter = ax.scatter(x_vals, y_vals, c=interact_vals,
                             cmap="RdYlGn_r", alpha=0.5, s=15)
        plt.colorbar(scatter, ax=ax, label=interaction_feature)
    else:
        ax.scatter(x_vals, y_vals, color="#378ADD", alpha=0.4, s=15)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value ({feature})")
    ax.set_title(f"SHAP Dependence: {feature}\n"
                 f"Dương = tăng churn probability, Âm = giảm",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Plot 4: Waterfall plot — test customer  ────────────────
def plot_shap_waterfall_single(explainer,X_sample: pd.DataFrame,customer_idx: int = 0,save_path: str = None,):
    customer = X_sample.iloc[[customer_idx]]

    # Tính SHAP values cho customer
    shap_vals_single = explainer.shap_values(customer)
    if isinstance(shap_vals_single, list):
        shap_vals_single = shap_vals_single[1]

    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.array(expected_value).mean()

    expected_value = float(expected_value)

    # Manual waterfall plot (compatible với tất cả phiên bản shap)
    shap_single = shap_vals_single[0].mean(axis=1)
    feature_names = list(X_sample.columns)
    feature_values = customer.values[0]

    # Sort theo |SHAP|, lấy top 10
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

    prob = explainer.predict(customer) if hasattr(explainer, "predict") else None
    title = f"SHAP Waterfall — Customer #{customer_idx}\n"
    title += f"Baseline: {expected_value:.2f} | "
    title += f"Predicted churn prob: {expected_value + shap_single.sum():.2f}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── ROI Calculator ────────────────────────────────────────────────────────
def compute_roi_table(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    avg_monthly_revenue: float = 65.0,
    avg_tenure_lost: float = 12.0,
    retention_cost: float = 50.0,
    retention_success_rate: float = 0.30,
) -> pd.DataFrame:
  
    y_pred = (y_prob >= threshold).astype(int)

    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    revenue_saved_per_customer = avg_monthly_revenue * avg_tenure_lost
    cost_per_tp = retention_cost
    revenue_per_tp = revenue_saved_per_customer * retention_success_rate

    no_model_loss = (tp + fn) * revenue_saved_per_customer

    model_cost   = (tp + fp) * retention_cost
    model_saved  = tp * revenue_per_tp
    model_net    = model_saved - model_cost

    rows = [
        ("Customers đúng phát hiện churn (TP)",    tp,    f"+${tp * revenue_per_tp:,.0f}"),
        ("Customers báo nhầm (FP) — retention cost", fp,  f"-${fp * retention_cost:,.0f}"),
        ("Customers bỏ sót (FN) — revenue lost",    fn,  f"-${fn * revenue_saved_per_customer:,.0f}"),
        ("Tổng chi phí retention",                  tp+fp, f"-${model_cost:,.0f}"),
        ("Tổng revenue saved",                      "-",   f"+${model_saved:,.0f}"),
        ("Net benefit so với không dùng model",     "-",   f"+${model_net:,.0f}"),
        ("Revenue lost nếu không có model",         tp+fn, f"-${no_model_loss:,.0f}"),
    ]

    df = pd.DataFrame(rows, columns=["Item", "Count", "Amount"])
    return df, model_net
