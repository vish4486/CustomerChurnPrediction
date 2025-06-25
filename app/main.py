# app/main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_and_inspect_data, preprocess_data,visualize_correlations
from src.model_train import train_models, evaluate_model_advanced
from src.hyperparameter_tuning import (
    tune_logistic_regression, tune_random_forest, tune_xgboost, tune_ann
)
from src.evaluate import plot_model_performance
from src.model_utils import save_sklearn_model, save_keras_model,save_model_metrics_table,export_model_metrics_latex





def main():
    # Step 1: Load + Preprocess
    df = load_and_inspect_data("data/raw/BankChurners.csv")
    visualize_correlations(df, save=True)

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
    tuned_results = []

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
    save_sklearn_model(best_lr, "logistic_model.pkl")
    save_sklearn_model(best_rf, "random_forest_model.pkl")
    save_sklearn_model(best_xgb, "xgboost_model.pkl")
    save_keras_model(best_ann, "ann_model.keras")

    # Step 6: Optional visual comparison
    plot_model_performance(tuned_results)
    export_model_metrics_latex(tuned_results)



if __name__ == "__main__":
    main()
