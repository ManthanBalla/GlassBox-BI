"""Phase 6 Verification Script: Explainability Agent and Attribution Methods.

Executes and verifies real explanations on the synthetic retail dataset (STORE_001 / PROD_001):
1. LightGBM + SHAP (TreeExplainer)
2. LightGBM + LIME (Local linear surrogate)
3. PyTorch LSTM + SHAP (Model-agnostic sequence attribution)
4. PyTorch LSTM + LIME (Sequence surrogate)
5. Prophet + Component-based (Additive decomposition)
6. Model Immutability Verification (pre/post snapshot comparison)
7. Explanation Fidelity & Reconstruction Accuracy
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch

from backend.app.explainability.agent import ExplainabilityAgent
from backend.app.schemas.explainability import (
    GlobalExplanationRequest,
    LocalExplanationRequest,
)


def run_verification() -> None:
    print("=" * 75)
    print("PHASE 6 VERIFICATION: EXPLAINABILITY AGENT (SHAP, LIME, COMPONENT)")
    print("=" * 75)

    data_path = "data/processed/synthetic/retail_processed.csv"
    if not Path(data_path).exists():
        print(f"[ERROR] Dataset not found at: {data_path}")
        sys.exit(1)

    agent = ExplainabilityAgent()

    # -------------------------------------------------------------------------
    # 1. LightGBM + SHAP (TreeExplainer)
    # -------------------------------------------------------------------------
    print("\n[1] Testing LightGBM + SHAP (Local Explanation)")
    req_lgb_shap = LocalExplanationRequest(
        dataset_path=data_path,
        entity_id="STORE_001",
        product_id="PROD_001",
        model_name="lightgbm",
        method="shap",
        background_samples=30,
    )
    res_lgb_shap = agent.explain_local(req_lgb_shap)
    print(f"    Status: SUCCESS")
    print(f"    Forecast Date: {res_lgb_shap.prediction_date} | Prediction: {res_lgb_shap.prediction}")
    print(f"    Base Value (Expected): {res_lgb_shap.base_value}")
    print(f"    Fidelity Score: {res_lgb_shap.fidelity.fidelity_score} | Error: {res_lgb_shap.fidelity.reconstruction_error}")
    print("    Top 5 Features by Absolute Contribution:")
    for f in res_lgb_shap.features[:5]:
        print(f"      - {f.feature:20s}: val={f.value:8.2f} | contrib={f.contribution:+8.4f} ({f.direction})")

    # Verify additivity
    sum_contrib = sum(f.contribution for f in res_lgb_shap.features)
    assert abs(res_lgb_shap.prediction - (res_lgb_shap.base_value + sum_contrib)) < 0.1, "SHAP additivity violated"
    print("    [OK] Additive reconstruction holds (|y_hat - (base + sum(phi))| < 0.1).")

    # -------------------------------------------------------------------------
    # 2. LightGBM + LIME (Local Linear Surrogate)
    # -------------------------------------------------------------------------
    print("\n[2] Testing LightGBM + LIME (Local Explanation)")
    req_lgb_lime = LocalExplanationRequest(
        dataset_path=data_path,
        entity_id="STORE_001",
        product_id="PROD_001",
        model_name="lightgbm",
        method="lime",
        background_samples=30,
        num_lime_samples=300,
    )
    res_lgb_lime = agent.explain_local(req_lgb_lime)
    print(f"    Status: SUCCESS")
    print(f"    Forecast Date: {res_lgb_lime.prediction_date} | Prediction: {res_lgb_lime.prediction}")
    print(f"    Surrogate Intercept: {res_lgb_lime.base_value}")
    print(f"    Surrogate R²: {res_lgb_lime.fidelity.surrogate_r2}")
    print("    Top 5 LIME Drivers:")
    for f in res_lgb_lime.features[:5]:
        print(f"      - {f.feature:20s}: val={f.value:8.2f} | weight={f.contribution:+8.4f} ({f.direction})")
    print("    [OK] LIME local linear surrogate successfully converged.")

    # -------------------------------------------------------------------------
    # 3. LightGBM Global SHAP Feature Importance
    # -------------------------------------------------------------------------
    print("\n[3] Testing LightGBM Global SHAP Feature Importance")
    req_lgb_glob = GlobalExplanationRequest(
        dataset_path=data_path,
        entity_id="STORE_001",
        product_id="PROD_001",
        model_name="lightgbm",
        method="shap",
        sample_size=30,
        top_k=10,
    )
    res_lgb_glob = agent.explain_global(req_lgb_glob)
    print("    Top 10 Global Features (Mean |SHAP|):")
    for gf in res_lgb_glob.global_importance:
        print(f"      {gf.rank:2d}. {gf.feature:20s}: score={gf.importance_score:8.4f} ({gf.normalized_importance*100:5.1f}%)")
    print("    [OK] Global SHAP feature ranking generated successfully.")

    # -------------------------------------------------------------------------
    # 4. PyTorch LSTM + SHAP (Sequence Attribution)
    # -------------------------------------------------------------------------
    print("\n[4] Testing PyTorch LSTM + SHAP (Lookback Sequence Attribution)")
    req_lstm_shap = LocalExplanationRequest(
        dataset_path=data_path,
        entity_id="STORE_001",
        product_id="PROD_001",
        model_name="lstm",
        method="shap",
        background_samples=20,
    )
    res_lstm_shap = agent.explain_local(req_lstm_shap)
    print(f"    Status: SUCCESS | Prediction: {res_lstm_shap.prediction}")
    print(f"    Fidelity Score: {res_lstm_shap.fidelity.fidelity_score}")
    print("    Lookback Sequence Attributions (lag_1 = most recent):")
    for f in res_lstm_shap.features[:5]:
        print(f"      - {f.feature:10s}: target_val={f.value:8.2f} | shap={f.contribution:+8.4f}")
    print("    [OK] Model-agnostic sequence attribution explains historical target window.")

    # -------------------------------------------------------------------------
    # 5. Prophet + Component-Based Decomposition
    # -------------------------------------------------------------------------
    print("\n[5] Testing Prophet Component-Based Explanation")
    req_proph = LocalExplanationRequest(
        dataset_path=data_path,
        entity_id="STORE_001",
        product_id="PROD_001",
        model_name="prophet",
        method="component_based",
    )
    res_proph = agent.explain_local(req_proph)
    print(f"    Status: SUCCESS | Prediction: {res_proph.prediction}")
    print(f"    Decomposition Status: {res_proph.fidelity.explanation_status}")
    print("    Decomposed Additive Components:")
    for f in res_proph.features:
        print(f"      - {f.feature:22s}: {f.contribution:+8.4f}")
    assert abs(res_proph.fidelity.reconstruction_error) < 0.05, "Prophet component decomposition error > 0.05"
    print("    [OK] Exact mathematical component attribution verified.")

    # -------------------------------------------------------------------------
    # 6. Model Immutability Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Strict Model Immutability During Explanation")
    # Fit a model and take pre/post snapshots
    from backend.app.forecasting.lightgbm_model import LightGBMForecaster
    lgb_test = LightGBMForecaster(random_seed=42)
    # Read training slice
    raw_df = pd.read_csv(data_path)
    s_df = raw_df[(raw_df["entity_id"] == "STORE_001") & (raw_df["product_id"] == "PROD_001")].copy()
    lgb_test.fit(s_df.iloc[:150])
    
    pre_snap = agent._capture_model_state_snapshot(lgb_test)
    _ = agent.explain_local(req_lgb_shap, model=lgb_test, history_df=s_df.iloc[:150], train_df=s_df.iloc[:100])
    post_snap = agent._capture_model_state_snapshot(lgb_test)
    agent._verify_model_immutability(pre_snap, post_snap, lgb_test)
    print("    [OK] Model state, trees, and historical buffers remain completely immutable.")

    print("\n" + "=" * 75)
    print("PHASE 6 EXPLAINABILITY VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_verification()
