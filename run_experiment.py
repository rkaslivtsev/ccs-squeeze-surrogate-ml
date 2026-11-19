"""
Physics-Informed Surrogate Modeling for Near-Wellbore Salt Precipitation
and Preventive Squeeze Treatments in CO2 Injection Wells.

Generates a 2,500-scenario Latin Hypercube synthetic dataset coupling
evaporative halite precipitation, capillary backflow, Verma-Pruess
permeability impairment, and temperature-dependent Freundlich sorption,
then benchmarks MLR, Random Forest, and XGBoost surrogates with SHAP
interpretability analysis.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import qmc
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")

# Publication-quality Matplotlib configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_compute_device():
    """Check whether a CUDA-capable GPU is available for XGBoost."""
    try:
        X_dummy = np.random.rand(10, 3)
        y_dummy = np.random.rand(10)
        test_model = XGBRegressor(n_estimators=2, tree_method="hist", device="cuda")
        test_model.fit(X_dummy, y_dummy)
        print("[INFO] CUDA-capable GPU detected; enabling GPU acceleration for XGBoost.")
        return "cuda"
    except Exception:
        print("[INFO] No CUDA GPU detected; falling back to multi-threaded CPU execution.")
        return "cpu"


def physics_forward_model(df_params):
    """
    Coupled 1D radial near-wellbore physics model for halite precipitation
    and dual-function preventive squeeze treatments.

    Components:
    1. Moisture balance and counter-current capillary brine backflow.
    2. Modified Verma-Pruess permeability impairment model.
    3. Temperature-dependent Freundlich adsorption-desorption isotherm.
    """
    phi0 = df_params["Porosity"].values            # Initial porosity (fraction)
    k0 = df_params["Permeability_mD"].values       # Initial permeability (mD)
    Cs = df_params["Salinity_gL"].values           # Formation brine salinity (g/L = kg/m3)
    T_C = df_params["Temperature_C"].values        # Reservoir temperature (°C)
    Q_inj = df_params["CO2_Rate_Mtyr"].values      # CO2 injection rate (Mt/yr)
    Kf0 = df_params["Freundlich_Kf0"].values       # Reference Freundlich constant at 40 °C
    n_fr = df_params["Freundlich_n"].values        # Freundlich nonlinearity exponent (-)
    C_slug = df_params["Squeeze_Conc_wt"].values   # Composite squeeze concentration (wt.%)

    # Physical and thermodynamic constants
    rho_halite = 2165.0      # Crystalline NaCl density (kg/m3)
    R_gas = 8.314            # Universal gas constant (J/(mol*K))
    dH_ads = -12500.0        # Exothermic enthalpy of adsorption (J/mol)
    T0_K = 40.0 + 273.15     # Reference temperature (K)
    T_K = T_C + 273.15
    MIC_mgL = 10.0           # Minimum Inhibitor Concentration (mg/L)

    # 1. Immobile residual water saturation after scCO2 breakthrough
    Swr = np.clip(0.36 - 0.35 * phi0 - 0.022 * np.log(k0 / 10.0), 0.12, 0.38)

    # 2. Temperature-adjusted Freundlich adsorption constant Kf(T)
    Kf_T = Kf0 * np.exp(-(dH_ads / R_gas) * (1.0 / T_K - 1.0 / T0_K))

    # 3. Baseline capillary backflow amplification factor (untreated well)
    beta_cap_0 = 1.0 + 3.1 * ((Cs / 300.0) ** 0.65) * ((0.5 / Q_inj) ** 0.35) * ((100.0 / k0) ** 0.15)
    beta_cap_0 = np.clip(beta_cap_0, 1.2, 6.5)

    # 4. Backflow suppression via adsorbed surface-active modifier
    eta_surf = np.where(
        C_slug > 0,
        (C_slug * Kf_T * 10.0) / (C_slug * Kf_T * 10.0 + 2.2),
        0.0
    )
    beta_cap = 1.0 + (beta_cap_0 - 1.0) * (1.0 - 0.88 * eta_surf)

    # 5. Solid halite pore-space saturation (Ss)
    Ss = beta_cap * (Swr * Cs) / rho_halite
    Ss = np.clip(Ss, 0.002, 0.45)

    # 6. Modified Verma-Pruess permeability impairment model (phi_c = 0.25 * phi0)
    porosity_ratio = np.clip(1.0 - (Ss / 0.75), 0.05, 0.999)
    tau_eff = 3.85 - 1.25 * eta_surf
    k_ratio = porosity_ratio ** tau_eff
    perm_impairment_pct = (1.0 - k_ratio) * 100.0

    # 7. Protected injection volume V_prot (kt CO2 before concentration drops below MIC)
    C_inj_mgL = C_slug * 10000.0  # Convert wt.% to mg/L
    Gamma_init = np.where(C_slug > 0, Kf_T * (C_inj_mgL ** (1.0 / n_fr)), 0.0)
    Gamma_mic = np.where(C_slug > 0, Kf_T * (MIC_mgL ** (1.0 / n_fr)), 0.0)
    Gamma_usable = np.maximum(Gamma_init - Gamma_mic, 0.0)

    V_prot_kt = np.where(
        C_slug > 0,
        24.0 * ((1.0 - phi0) * Gamma_usable * (n_fr ** 0.65)) /
        ((Q_inj ** 0.25) * (1.0 + 0.002 * Cs)) * ((k0 / 100.0) ** 0.12),
        0.0
    )

    # Add 2.5% Gaussian noise to emulate natural core-scale petrophysical variability
    rng = np.random.default_rng(42)
    noise_k = rng.normal(0.0, 0.025 * np.mean(perm_impairment_pct), size=len(df_params))
    noise_v = np.where(
        C_slug > 0,
        rng.normal(0.0, 0.025 * np.mean(V_prot_kt[C_slug > 0]), size=len(df_params)),
        0.0
    )

    perm_impairment_pct = np.clip(perm_impairment_pct + noise_k, 1.0, 95.0)
    V_prot_kt = np.clip(V_prot_kt + noise_v, 0.0, None)

    df_out = df_params.copy()
    df_out["Perm_Impairment_pct"] = perm_impairment_pct
    df_out["Protection_Volume_kt"] = V_prot_kt
    return df_out


def generate_lhs_dataset(n_samples=2500):
    """Generate synthetic reservoir scenarios via Latin Hypercube Sampling (LHS)."""
    bounds = {
        "Porosity": (0.10, 0.30),
        "Permeability_mD": (10.0, 500.0),
        "Salinity_gL": (50.0, 300.0),
        "Temperature_C": (40.0, 110.0),
        "CO2_Rate_Mtyr": (0.2, 2.0),
        "Freundlich_Kf0": (0.05, 0.50),
        "Freundlich_n": (1.5, 4.5),
        "Squeeze_Conc_wt": (0.0, 15.0)
    }

    sampler = qmc.LatinHypercube(d=len(bounds), seed=42)
    sample_unit = sampler.random(n=n_samples)

    l_bounds = [b[0] for b in bounds.values()]
    u_bounds = [b[1] for b in bounds.values()]
    sample_scaled = qmc.scale(sample_unit, l_bounds, u_bounds)

    # Assign 10% of scenarios as untreated baseline injectors (Squeeze_Conc_wt = 0)
    n_untreated = int(0.10 * n_samples)
    sample_scaled[:n_untreated, -1] = 0.0

    df_params = pd.DataFrame(sample_scaled, columns=list(bounds.keys()))
    return physics_forward_model(df_params)


def evaluate_models(X_train, X_test, y_train, y_test, target_name, device, use_log_transform=False):
    """Train and benchmark MLR, Random Forest, and XGBoost with 5-fold cross-validation."""
    y_fit_target = np.log1p(y_train) if use_log_transform else y_train

    base_models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "XGBoost (Hist)": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            tree_method="hist",
            device=device,
            base_score=float(np.mean(y_fit_target)),
            random_state=42
        )
    }

    models = {}
    for name, reg in base_models.items():
        if use_log_transform:
            models[name] = TransformedTargetRegressor(
                regressor=reg, func=np.log1p, inverse_func=np.expm1
            )
        else:
            models[name] = reg

    results = []
    trained_models = {}
    predictions = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        t0 = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        r2_tr = r2_score(y_train, y_pred_train)
        r2_te = r2_score(y_test, y_pred_test)
        rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_te = mean_absolute_error(y_test, y_pred_test)

        mask = y_test > 1.0
        mrpe_te = np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100.0

        results.append({
            "Target": target_name,
            "Model": name,
            "CV_R2_Mean": round(float(np.mean(cv_scores)), 4),
            "CV_R2_Std": round(float(np.std(cv_scores)), 4),
            "Train_R2": round(float(r2_tr), 4),
            "Test_R2": round(float(r2_te), 4),
            "Test_RMSE": round(float(rmse_te), 3),
            "Test_MAE": round(float(mae_te), 3),
            "Test_MRPE_pct": round(float(mrpe_te), 2),
            "Time_sec": round(train_time, 2)
        })

        trained_models[name] = model
        predictions[name] = y_pred_test

    return pd.DataFrame(results), trained_models, predictions


def plot_fig1_parity(y_test_k, preds_k, y_test_v, preds_v):
    """Figure 1: Parity plots comparing physics-based outputs and ML predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # Panel (a): Permeability impairment
    ax = axes[0]
    ax.scatter(y_test_k, preds_k["Linear Regression"], alpha=0.35, s=16, color="#95a5a6", label="Linear Regression")
    ax.scatter(y_test_k, preds_k["Random Forest"], alpha=0.45, s=18, color="#2980b9", label="Random Forest")
    ax.scatter(y_test_k, preds_k["XGBoost (Hist)"], alpha=0.65, s=20, color="#c0392b", label="XGBoost (Surrogate)")
    lims = [0, max(y_test_k.max(), preds_k["XGBoost (Hist)"].max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1.3, label="Ideal 1:1 Fit")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Physics-Based Permeability Impairment, $\\Delta k_{imp}$ (%)")
    ax.set_ylabel("ML Predicted Permeability Impairment, $\\Delta k_{imp}$ (%)")
    ax.set_title("(a) Near-Wellbore Permeability Impairment Prediction")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel (b): Squeeze protection lifetime (log-transformed target)
    ax = axes[1]
    ax.scatter(y_test_v, preds_v["Linear Regression"], alpha=0.35, s=16, color="#95a5a6", label="Linear Regression (Log)")
    ax.scatter(y_test_v, preds_v["Random Forest"], alpha=0.45, s=18, color="#2980b9", label="Random Forest (Log)")
    ax.scatter(y_test_v, preds_v["XGBoost (Hist)"], alpha=0.65, s=20, color="#16a085", label="XGBoost (Log-Surrogate)")
    lims_v = [0, max(y_test_v.max(), preds_v["XGBoost (Hist)"].max()) * 1.05]
    ax.plot(lims_v, lims_v, "k--", lw=1.3, label="Ideal 1:1 Fit")
    ax.set_xlim(lims_v)
    ax.set_ylim(lims_v)
    ax.set_xlabel("Physics-Based Squeeze Lifetime, $V_{prot}$ (kt CO$_2$)")
    ax.set_ylabel("ML Predicted Squeeze Lifetime, $V_{prot}$ (kt CO$_2$)")
    ax.set_title("(b) Preventive Squeeze Lifetime Prediction (Log-Target)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_Parity_Plots.png"))
    plt.close()


def plot_fig2_shap_summary(xgb_k, xgb_v, X_test):
    """Figure 2: Global SHAP summary (beeswarm) plots for XGBoost surrogates."""
    nice_names = [
        "Porosity ($\\phi_0$)",
        "Permeability ($k_0$)",
        "Brine Salinity ($C_s$)",
        "Temperature ($T$)",
        "CO$_2$ Injection Rate ($Q_{inj}$)",
        "Freundlich $K_{F,0}$",
        "Freundlich $n$",
        "Squeeze Conc. ($C_{slug}$)"
    ]
    X_disp = X_test.copy()
    X_disp.columns = nice_names

    inner_xgb_k = xgb_k.regressor_ if hasattr(xgb_k, "regressor_") else xgb_k
    explainer_k = shap.TreeExplainer(inner_xgb_k)
    shap_values_k = explainer_k.shap_values(X_test)

    inner_xgb_v = xgb_v.regressor_ if hasattr(xgb_v, "regressor_") else xgb_v
    explainer_v = shap.TreeExplainer(inner_xgb_v)
    shap_values_v = explainer_v.shap_values(X_test)

    fig = plt.figure(figsize=(14, 5.5))

    ax1 = fig.add_subplot(1, 2, 1)
    plt.sca(ax1)
    shap.summary_plot(shap_values_k, X_disp, plot_size=None, show=False)
    ax1.set_title("(a) SHAP Impact on Permeability Impairment ($\\Delta k_{imp}$)")
    ax1.set_xlabel("SHAP Value (Impact on $\\Delta k_{imp}$, %)")

    ax2 = fig.add_subplot(1, 2, 2)
    plt.sca(ax2)
    shap.summary_plot(shap_values_v, X_disp, plot_size=None, show=False)
    ax2.set_title("(b) SHAP Impact on Squeeze Lifetime ($\\ln(1 + V_{prot})$)")
    ax2.set_xlabel("SHAP Value (Impact on $\\ln(1 + V_{prot})$)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_SHAP_Summary.png"))
    plt.close()

    return shap_values_k, shap_values_v


def plot_fig3_shap_dependence(shap_values_k, shap_values_v, X_test):
    """Figure 3: SHAP dependence plots illustrating coupled physical interactions."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    shap.dependence_plot(
        "Salinity_gL",
        shap_values_k,
        X_test,
        interaction_index="Squeeze_Conc_wt",
        ax=axes[0],
        show=False
    )
    axes[0].set_xlabel("Formation Brine Salinity, $C_s$ (g/L)")
    axes[0].set_ylabel("SHAP Value for $\\Delta k_{imp}$ (%)")
    axes[0].set_title("(a) Salting-Out Risk vs. Composite Squeeze Protection")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    shap.dependence_plot(
        "Freundlich_Kf0",
        shap_values_v,
        X_test,
        interaction_index="Temperature_C",
        ax=axes[1],
        show=False
    )
    axes[1].set_xlabel("Freundlich Adsorption Constant, $K_{F,0}$")
    axes[1].set_ylabel("SHAP Value for $\\ln(1 + V_{prot})$")
    axes[1].set_title("(b) Inhibitor Retention Capacity vs. Reservoir Temperature")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_SHAP_Dependence.png"))
    plt.close()


def plot_fig4_decision_map(xgb_k, xgb_v):
    """Figure 4: Surrogate-generated engineering design maps for squeeze optimization."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    sal_grid = np.linspace(50, 300, 60)
    conc_grid = np.linspace(0, 15, 60)
    SS_mesh, CC_mesh = np.meshgrid(sal_grid, conc_grid)

    grid_df = pd.DataFrame({
        "Porosity": 0.20,
        "Permeability_mD": 120.0,
        "Salinity_gL": SS_mesh.ravel(),
        "Temperature_C": 70.0,
        "CO2_Rate_Mtyr": 1.0,
        "Freundlich_Kf0": 0.25,
        "Freundlich_n": 2.5,
        "Squeeze_Conc_wt": CC_mesh.ravel()
    })

    Z_imp = xgb_k.predict(grid_df).reshape(SS_mesh.shape)
    cf = axes[0].contourf(SS_mesh, CC_mesh, Z_imp, levels=16, cmap="RdYlBu_r")
    cs = axes[0].contour(SS_mesh, CC_mesh, Z_imp, levels=[15, 25, 35, 50], colors="black", linewidths=1.0)
    axes[0].clabel(cs, inline=True, fontsize=8, fmt="%.0f%%")
    cbar = fig.colorbar(cf, ax=axes[0])
    cbar.set_label("Predicted Permeability Impairment, $\\Delta k_{imp}$ (%)")
    axes[0].set_xlabel("Formation Brine Salinity, $C_s$ (g/L)")
    axes[0].set_ylabel("Composite Squeeze Concentration, $C_{slug}$ (wt.%)")
    axes[0].set_title("(a) Permeability Impairment Design Map ($k_0=120$ mD, $T=70^\\circ$C)")

    conc_line = np.linspace(0.5, 15.0, 50)
    temps = [45.0, 75.0, 105.0]
    colors = ["#2980b9", "#e67e22", "#c0392b"]

    for T_val, col in zip(temps, colors):
        line_df = pd.DataFrame({
            "Porosity": 0.20,
            "Permeability_mD": 120.0,
            "Salinity_gL": 200.0,
            "Temperature_C": T_val,
            "CO2_Rate_Mtyr": 1.0,
            "Freundlich_Kf0": 0.25,
            "Freundlich_n": 2.5,
            "Squeeze_Conc_wt": conc_line
        })
        v_pred = xgb_v.predict(line_df)
        axes[1].plot(conc_line, v_pred, lw=2.2, color=col, label=f"Reservoir $T = {int(T_val)}^\\circ$C")

    axes[1].set_xlabel("Composite Squeeze Concentration, $C_{slug}$ (wt.%)")
    axes[1].set_ylabel("Protected Injection Volume, $V_{prot}$ (kt CO$_2$)")
    axes[1].set_title("(b) Squeeze Lifetime vs. Temperature ($C_s=200$ g/L)")
    axes[1].legend(frameon=True, facecolor="white")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_Squeeze_Optimization_Map.png"))
    plt.close()


def main():
    print("=" * 68)
    print("PHYSICS-INFORMED ML SURROGATE MODELING FOR CCS SQUEEZE TREATMENTS")
    print("=" * 68)

    device = detect_compute_device()

    print("\n[1/5] Generating physics-informed synthetic dataset (2,500 LHS scenarios)...")
    df = generate_lhs_dataset(n_samples=2500)
    dataset_path = os.path.join(OUTPUT_DIR, "synthetic_ccs_squeeze_dataset.csv")
    df.to_csv(dataset_path, index=False)
    print(f"      Dataset saved to: {dataset_path}")

    feature_cols = [
        "Porosity", "Permeability_mD", "Salinity_gL", "Temperature_C",
        "CO2_Rate_Mtyr", "Freundlich_Kf0", "Freundlich_n", "Squeeze_Conc_wt"
    ]
    X = df[feature_cols]
    y_k = df["Perm_Impairment_pct"]
    y_v = df["Protection_Volume_kt"]

    X_train, X_test, yk_train, yk_test, yv_train, yv_test = train_test_split(
        X, y_k, y_v, test_size=0.20, random_state=42
    )

    print("\n[2/5] Training surrogate models for permeability impairment (Perm_Impairment_pct)...")
    metrics_k, models_k, preds_k = evaluate_models(
        X_train, X_test, yk_train, yk_test, "Perm_Impairment (%)", device, use_log_transform=False
    )

    print("[3/5] Training log-target surrogates for squeeze lifetime (Protection_Volume_kt)...")
    metrics_v, models_v, preds_v = evaluate_models(
        X_train, X_test, yv_train, yv_test, "Protection_Volume (kt)", device, use_log_transform=True
    )

    all_metrics = pd.concat([metrics_k, metrics_v], ignore_index=True)
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics_table.csv")
    all_metrics.to_csv(metrics_path, index=False)

    print("\n" + "=" * 68)
    print("MODEL BENCHMARK SUMMARY (5-FOLD CV & BLIND TEST SET, n = 500)")
    print("=" * 68)
    print(all_metrics.to_string(index=False))
    print("=" * 68)

    print("\n[4/5] Computing SHAP TreeExplainer values and exporting 300 DPI figures...")
    plot_fig1_parity(yk_test, preds_k, yv_test, preds_v)
    shap_k, shap_v = plot_fig2_shap_summary(
        models_k["XGBoost (Hist)"], models_v["XGBoost (Hist)"], X_test
    )
    plot_fig3_shap_dependence(shap_k, shap_v, X_test)
    plot_fig4_decision_map(models_k["XGBoost (Hist)"], models_v["XGBoost (Hist)"])

    print(f"\n[5/5] Complete. All outputs saved to: ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()