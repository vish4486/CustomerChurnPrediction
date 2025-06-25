from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from sklearn.utils import class_weight
import numpy as np
from collections import Counter


def tune_logistic_regression(X, y):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        param_grid,
        scoring='f1_weighted',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def tune_random_forest(X, y):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'max_features': ['sqrt']
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid,
        scoring='f1_weighted',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def tune_xgboost(X, y):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
    }
    counter = Counter(y)
    scale_pos_weight = counter[0] / counter[1]

    grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight),
        param_grid,
        scoring='f1_weighted',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def build_ann_model(hp, input_shape):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', 16, 64, step=16),
                    activation=hp.Choice('act1', ['relu', 'tanh']),
                    input_shape=(input_shape,)))
    model.add(Dense(units=hp.Int('units2', 16, 64, step=16),
                    activation=hp.Choice('act2', ['relu', 'tanh'])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('lr', [0.001, 0.0005])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def tune_ann(X, y):
    tuner = kt.RandomSearch(
        lambda hp: build_ann_model(hp, X.shape[1]),
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='ann_tuning',
        project_name='churn_ann'
    )
    tuner.search(X, y, epochs=20, validation_split=0.2, verbose=0)
    best_hps = tuner.get_best_hyperparameters(1)[0]
    model = build_ann_model(best_hps, X.shape[1])
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_dict = dict(enumerate(class_weights))
    model.fit(X, y, epochs=20, batch_size=32, verbose=0, class_weight=class_weights_dict)
    return model, best_hps
