import time
import numpy as np
import joblib
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def evaluate_model_advanced(name, model, X_train, y_train, X_test, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    y_pred = model.predict(X_test)

    if hasattr(y_pred, "toarray"):
        y_pred = y_pred.toarray()

    if y_pred.ndim > 1 or y_pred.dtype != int:
        y_pred = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    train_time = end - start

    return {
        'Model': name,
        'Balanced Accuracy': round(bal_acc, 3),
        'F1 Score': round(f1, 3),
        'Precision': round(precision, 3),
        'Recall (TPR)': round(recall, 3),
        'Specificity (TNR)': round(TN / (TN + FP), 3),
        'FPR': round(FP / (FP + TN), 3),
        'FNR': round(FN / (FN + TP), 3),
        'ROC AUC': round(auc, 3),
        'Training Time (s)': round(train_time, 3)
    }


def train_models(X_train, y_train, X_test, y_test):
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]

    results = []

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)

    results.append(evaluate_model_advanced("Logistic Regression", lr, X_train, y_train, X_test, y_test))
    results.append(evaluate_model_advanced("Random Forest", rf, X_train, y_train, X_test, y_test))
    results.append(evaluate_model_advanced("XGBoost", xgb, X_train, y_train, X_test, y_test))

    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    ann = Sequential([
        Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    ann.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, class_weight=class_weights_dict)
    end = time.time()

    y_pred_probs = ann.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    ann_bal_acc = balanced_accuracy_score(y_test, y_pred)
    ann_f1 = f1_score(y_test, y_pred, average='weighted')
    ann_precision = precision_score(y_test, y_pred, average='weighted')
    ann_recall = recall_score(y_test, y_pred, average='weighted')
    ann_auc = roc_auc_score(y_test, y_pred)

    results.append({
        'Model': 'ANN (class_weight)',
        'Balanced Accuracy': round(ann_bal_acc, 3),
        'F1 Score': round(ann_f1, 3),
        'Precision': round(ann_precision, 3),
        'Recall (TPR)': round(ann_recall, 3),
        'Specificity (TNR)': round(TN / (TN + FP), 3),
        'FPR': round(FP / (FP + TN), 3),
        'FNR': round(FN / (FN + TP), 3),
        'ROC AUC': round(ann_auc, 3),
        'Training Time (s)': round(end - start, 3)
    })

    return results, lr, rf, xgb, ann


def save_models(lr, rf, xgb, ann):
    joblib.dump(lr, "models/logistic_model.pkl")
    joblib.dump(rf, "models/random_forest_model.pkl")
    joblib.dump(xgb, "models/xgboost_model.pkl")
    ann.save("models/ann_model.keras")
