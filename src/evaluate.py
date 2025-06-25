import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.model_utils import save_plot


def display_confusion_matrix(y_true, y_pred, model_name='Model'):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")

    # Save image instead of show (for report)
    from src.model_utils import save_plot
    save_plot(plt.gcf(), f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", subfolder="plots")
    plt.close()


def plot_model_performance(results):
    df = pd.DataFrame(results)
    metrics = ['Balanced Accuracy', 'F1 Score', 'Precision', 'Recall (TPR)', 'ROC AUC']
    #plt.figure(figsize=(12, 6))
    for metric in metrics:
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y=metric, data=df)
        plt.title(f'Model Comparison - {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()
        filename = f"{metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_comparison.png"
        save_plot(fig, filename)


def plot_feature_importance(model, feature_names, model_name="Model"):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=df_importance.head(15), ax=ax)
        ax.set_title(f"{model_name} - Top Feature Importances")
        save_plot(fig, f"{model_name.lower().replace(' ', '_')}_feature_importance.png", subfolder="plots")
