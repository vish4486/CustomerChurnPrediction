# app/main.py
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_and_inspect_data, preprocess_data,visualize_correlations,visualize_churn_distribution,perform_additional_eda
from src.model_train import train_models, evaluate_model_advanced
from src.hyperparameter_tuning import (
    tune_logistic_regression, tune_random_forest, tune_xgboost, tune_ann
)
from src.evaluate import plot_model_performance,display_confusion_matrix,plot_feature_importance
from src.model_utils import save_sklearn_model, save_keras_model,save_model_metrics_table,export_model_metrics_latex
import pandas as pd




def main():
    # Step 1: Load + Preprocess
    df = load_and_inspect_data("data/raw/BankChurners.csv")
    #visualize_correlations(df, save=True)
    visualize_churn_distribution(df, save=True)
    visualize_correlations(df, save=True)
    perform_additional_eda(df)

    X_train, X_test, y_train, y_test, preprocessor, columns = preprocess_data(df)

    # Step 2: Train baseline models
    baseline_results, lr, rf, xgb, ann = train_models(X_train, y_train, X_test, y_test)
    print("\nBASELINE MODEL RESULTS")
    for res in baseline_results:
        print(res)

    # Step 3: Hyperparameter Tuning
    print("\n--- Hyperparameter Tuning ---")
    best_lr, lr_params = tune_logistic_regression(X_train, y_train)
    best_rf, rf_params = tune_random_forest(X_train, y_train)
    best_xgb, xgb_params = tune_xgboost(X_train, y_train)
    best_ann, ann_hps = tune_ann(X_train, y_train)

    print("Best LR Params:", lr_params)
    print("Best RF Params:", rf_params)
    print("Best XGB Params:", xgb_params)
    print("Best ANN Hyperparameters:", ann_hps.values)

    # Step 4: Evaluate tuned models
    tuned_results = []
    #tuned_results.append(evaluate_model_advanced("Tuned Logistic Regression", best_lr, X_train, y_train, X_test, y_test))
    #tuned_results.append(evaluate_model_advanced("Tuned Random Forest", best_rf, X_train, y_train, X_test, y_test))
    #tuned_results.append(evaluate_model_advanced("Tuned XGBoost", best_xgb, X_train, y_train, X_test, y_test))
    #tuned_results.append(evaluate_model_advanced("Tuned ANN", best_ann, X_train, y_train, X_test, y_test))
    

    tuned_results.append({
    **evaluate_model_advanced("Tuned Logistic Regression", best_lr, X_train, y_train, X_test, y_test),
    "Best Parameters": str(lr_params)
})
    tuned_results.append({
    **evaluate_model_advanced("Tuned Random Forest", best_rf, X_train, y_train, X_test, y_test),
    "Best Parameters": str(rf_params)
})

    tuned_results.append({
    **evaluate_model_advanced("Tuned XGBoost", best_xgb, X_train, y_train, X_test, y_test),
    "Best Parameters": str(xgb_params)
})

    tuned_results.append({
    **evaluate_model_advanced("Tuned ANN", best_ann, X_train, y_train, X_test, y_test),
    "Best Parameters": str(ann_hps.values)
})


    print("\nTUNED MODEL RESULTS")
    for res in tuned_results:
        print(res)

    # Step 5: Save tuned models
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save with timestamped filenames
    save_sklearn_model(best_lr, f"logistic_model_{timestamp}.pkl")
    save_sklearn_model(best_rf, f"random_forest_model_{timestamp}.pkl")
    save_sklearn_model(best_xgb, f"xgboost_model_{timestamp}.pkl")
    save_keras_model(best_ann, f"ann_model_{timestamp}.keras")

    # Automatically save best model and its metadata
    best_model_result = max(tuned_results, key=lambda x: x["ROC AUC"])
    best_model_name = best_model_result["Model"]
    print(f"\nAutomatically Selected Best Model: {best_model_name} based on ROC AUC")
    if "Logistic Regression" in best_model_name:
        save_sklearn_model(best_lr, "best_model.pkl")
    elif "Random Forest" in best_model_name:
        save_sklearn_model(best_rf, "best_model.pkl")
    elif "Tuned XGBoost" in best_model_name:
        save_sklearn_model(best_xgb, "best_model.pkl")
    elif "ANN" in best_model_name:
        save_keras_model(best_ann, "best_model.keras")


    metadata = {
    "best_model_type": best_model_name,
    "timestamp": timestamp,
    "metrics": best_model_result
    }

    # Add top features if tree-based model
    if "Tuned XGBoost" in best_model_name:
        feature_importances = best_xgb.feature_importances_
        X_train_df = pd.DataFrame(X_train, columns=columns)
        top_features = pd.Series(feature_importances, index=X_train_df.columns)\
                        .sort_values(ascending=False).head(10).index.tolist()
        metadata["top_features"] = top_features
    elif "Tuned Random Forest" in best_model_name:
        feature_importances = best_rf.feature_importances_
        X_train_df = pd.DataFrame(X_train, columns=columns)
        top_features = pd.Series(feature_importances, index=X_train_df.columns)\
                        .sort_values(ascending=False).head(10).index.tolist()
        metadata["top_features"] = top_features
    
    metadata["top_features"] = top_features
    with open("models/best_model_meta.json", "w") as f:
        json.dump(metadata, f, indent=4)
    

    # Step 6: Optional visual comparison
    plot_model_performance(tuned_results)
    export_model_metrics_latex(tuned_results)

    # Step 7: Confusion Matrices
    display_confusion_matrix(y_test, best_rf.predict(X_test), model_name="Random Forest")
    display_confusion_matrix(y_test, best_xgb.predict(X_test), model_name="XGBoost")

    # Step 8: Feature Importances
    plot_feature_importance(best_rf, columns, model_name="Random Forest")
    plot_feature_importance(best_xgb, columns, model_name="XGBoost")



if __name__ == "__main__":
    main()
