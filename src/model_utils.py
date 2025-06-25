import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pandas.plotting import table


def save_sklearn_model(model, filename):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{filename}")


def load_sklearn_model(filename):
    return joblib.load(f"models/{filename}")


def save_keras_model(model, filename):
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{filename}")


def load_keras_model(filename):
    return load_model(f"models/{filename}")


def save_plot(fig, filename, subfolder="plots"):
    os.makedirs(f"results/{subfolder}", exist_ok=True)
    fig.savefig(f"results/{subfolder}/{filename}", bbox_inches='tight')
    plt.close(fig)


def save_model_metrics_table(results, filename="model_metrics_table.png"):
    os.makedirs("results/plots", exist_ok=True)
    df = pd.DataFrame(results)

    # Reorder & rename for LaTeX-style table
    df = df[['Model', 'Best Parameters', 'Balanced Accuracy', 'F1 Score', 'ROC AUC', 'Training Time (s)']]
    df.columns = ['Model', 'Best Parameters', 'Balanced Accuracy', 'F1 Score', 'ROC AUC', r'$\mu_t$ (s)']

    fig, ax = plt.subplots(figsize=(12, len(df) * 0.75 + 1))
    ax.axis('off')
    tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.2] * len(df.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    plt.title("Table 1: Performance Metrics of Tuned Models", fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig("results/plots/" + filename, bbox_inches='tight', dpi=300)
    plt.close(fig)


def export_model_metrics_latex(results, filename="model_metrics_table.tex", subfolder="plots"):
    import os
    os.makedirs(f"results/{subfolder}", exist_ok=True)

    # Structure the DataFrame
    df = pd.DataFrame(results)
    df = df[['Model', 'Best Parameters', 'Balanced Accuracy', 'F1 Score', 'ROC AUC', 'Training Time (s)']]

    # LaTeX formatting
    latex_code = r"""\begin{table}[h!]
\centering
\caption{Performance Metrics of Tuned Models}
\label{tab:model_performance}
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Best Parameters} & \textbf{Balanced Accuracy} & \textbf{F1 Score} & \textbf{ROC AUC} & $\mu_t$ (s) \\
\midrule
"""
    for _, row in df.iterrows():
        model = row['Model']
        params = str(row['Best Parameters']).replace('_', r'\_').replace(',', r',\\')
        line = f"{model} & {params} & {row['Balanced Accuracy']} & {row['F1 Score']} & {row['ROC AUC']} & {row['Training Time (s)']} \\\\\n"
        latex_code += line

    latex_code += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(f"results/{subfolder}/{filename}", "w") as f:
        f.write(latex_code)
