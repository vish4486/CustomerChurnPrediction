import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.model_utils import save_plot
 

def display_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()


def plot_model_performance(results):
    df = pd.DataFrame(results)
    metrics = ['Balanced Accuracy', 'F1 Score', 'Precision', 'Recall (TPR)', 'ROC AUC']

    #plt.figure(figsize=(12, 6))
    for metric in metrics:
        fig=plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y=metric, data=df)
        plt.title(f'Model Comparison - {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()
        filename = f"{metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_comparison.png"
        save_plot(fig, filename)
