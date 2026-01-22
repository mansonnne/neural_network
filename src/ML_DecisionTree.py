import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, validation_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# ========== 0. Функция загрузки данных ==========
def load_data(filepath='data.csv'):
    df = pd.read_csv(filepath)
    print(f"Данные загружены. Размер: {df.shape}")
    print("==========  Первые 5 объектов ==============")
    print(df.head(5))
    return df

# ========== 1. Функция разведочного анализа ==========
def perform_eda(df, target_col='target'):
    eda_info = {
        'shape': df.shape,
        'dtypes': df.dtypes,
        'missing_values': df.isnull().sum(),
        'class_distribution': df[target_col].value_counts(normalize=True) if target_col in df.columns else None
    }
    print(f"Распределение классов:\n{eda_info['class_distribution']}")
    print(f"Пропуски:\n{eda_info['missing_values'][eda_info['missing_values'] > 0]}")
    
    return eda_info

# ========== 2. Функция предобработки данных ==========
def preprocess_data(df, target_col='target'):
    # Удаление строк с 50% пропусками (можно заменить на impute)
    df_clean = df.dropna(axis=1, thresh=int(df.shape[0]*0.5))
    # Разделение на признаки и целевую переменную
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    feature_names = X.columns.tolist()
    
    # Конвертация категориальных признаков (если есть)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        feature_names = X.columns.tolist()
    
    print(f"После предобработки: X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y, feature_names

# ========== 3. Функция разделения данных ==========
def split_data_stratified(X, y, test_size=0.2, val_size=0.2, random_state=42):
    # Сначала отделяем test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Затем train и val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=random_state
    )
    
    print(f"Размеры: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Распределение в train: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
    print(f"Распределение в val: {pd.Series(y_val).value_counts(normalize=True).to_dict()}")
    print(f"Распределение в test: {pd.Series(y_test).value_counts(normalize=True).to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ========== 4. Функция масштабирования признаков ==========
def scale_features(X_train, X_val, X_test, scaler_type='standard'):
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type должен быть 'standard' или 'minmax'")
    
    # Fit только на train, transform на всех

    # Заполняем
    imputer = SimpleImputer(strategy='mean')
    X_train_filled = imputer.fit_transform(X_train)
    X_val_filled = imputer.transform(X_val)
    X_test_filled = imputer.transform(X_test)

    # Масштабируем
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_val_scaled = scaler.transform(X_val_filled)
    X_test_scaled = scaler.transform(X_test_filled)
    
    print(f"Масштабирование ({scaler_type}) выполнено")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# ========== 5. Функция балансировки классов ==========
def balance_classes(X_train, y_train, balance_method='smote', random_state=42):
    """
    Балансировка классов в тренировочных данных
    Возвращает: сбалансированные X_train_bal, y_train_bal
    """
    if balance_method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    elif balance_method == 'none':
        X_train_bal, y_train_bal = X_train, y_train
    else:
        raise ValueError("balance_method должен быть 'smote' или 'none'")
    
    print(f"Балансировка ({balance_method}): было {X_train.shape[0]}, стало {X_train_bal.shape[0]}")
    print(f"Новое распределение: {pd.Series(y_train_bal).value_counts().to_dict()}")
    
    return X_train_bal, y_train_bal

# ========== 6. Функция обучения и подбора гиперпараметров ==========
def train_decision_tree(X_train, y_train, X_val=None, y_val=None, cv_folds=5, random_state=42):
    # param_grid = {
    #     'max_depth': [1, 2, 3, 4, 5, 6, 7, 10, 15, 20, None],
    #     'min_samples_split': [2, 3, 5,8,11,15],
    #     'min_samples_leaf': [1, 2, 4, 8],
    #     'criterion': ['gini', 'entropy']
    # }
    test_param_grid = {
        'max_depth': [1, 2, 3, 4, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['gini', 'entropy']
    }
    dt_model = DecisionTreeClassifier(random_state=random_state)
    
    # Кросс-валидация (стратифицированная для несбалансированных данных)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
    #    dt_model, param_grid, 
        dt_model, test_param_grid, 
        cv=cv, 
        scoring='f1',  # F1 лучше для несбалансированных классов
        n_jobs=-1,
        verbose=1
    )
    
    print("Запуск GridSearch...")
    grid_search.fit(X_train, y_train)
    
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучший F1-score (кросс-валидация): {grid_search.best_score_:.4f}")
    
    # Построение валидационной кривой по max_depth
    param_range = np.arange(1, 31)
    train_scores, val_scores = validation_curve(
        DecisionTreeClassifier(random_state=random_state, **{k: v for k, v in grid_search.best_params_.items() if k != 'max_depth'}),
        X_train, y_train,
        param_name='max_depth',
        param_range=param_range,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    
    val_curve_data = {
        'param_range': param_range,
        'train_scores_mean': np.mean(train_scores, axis=1),
        'train_scores_std': np.std(train_scores, axis=1),
        'val_scores_mean': np.mean(val_scores, axis=1),
        'val_scores_std': np.std(val_scores, axis=1)
    }
    
    return grid_search.best_estimator_, grid_search, val_curve_data

# ========== 7. Оценки модели ==========
def evaluate_model(model, X_val, y_val, X_train=None, y_train=None, model_name="Decision Tree"):
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_val, y_pred)
    }
    
    # ROC-AUC и PR-AUC
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_val, y_pred_proba)
        # Кривые для визуализации
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_pred_proba)
        metrics['roc_curve'] = (fpr, tpr)
        metrics['pr_curve'] = (precision_curve, recall_curve)
    
    # Оценка на тренировочных данных (если переданы)
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['train_f1'] = f1_score(y_train, y_train_pred, zero_division=0)
    
    # Вывод метрик
    print(f"\n=== Метрики {model_name} ===")
    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        if metric_name in metrics:
            print(f"{metric_name}: {metrics[metric_name]:.4f}")
    
    print(f"\nМатрица ошибок:\n{metrics['confusion_matrix']}")
    
    return metrics

# ========== 8. Функция визуализации валидационной кривой ==========
def plot_validation_curve(val_curve_data, best_max_depth=None):
    param_range = val_curve_data['param_range']
    
    plt.figure(figsize=(10, 6))
    plt.title("Validation Curve: Decision Tree (max_depth)")
    plt.xlabel("max_depth")
    plt.ylabel("F1 Score")
    plt.ylim(0.0, 1.1)
    
    plt.plot(param_range, val_curve_data['train_scores_mean'], label="Training score", color="blue", lw=2)
    plt.fill_between(param_range, 
                     val_curve_data['train_scores_mean'] - val_curve_data['train_scores_std'],
                     val_curve_data['train_scores_mean'] + val_curve_data['train_scores_std'],
                     alpha=0.2, color="blue")
    
    plt.plot(param_range, val_curve_data['val_scores_mean'], label="Cross-validation score", color="green", lw=2)
    plt.fill_between(param_range,
                     val_curve_data['val_scores_mean'] - val_curve_data['val_scores_std'],
                     val_curve_data['val_scores_mean'] + val_curve_data['val_scores_std'],
                     alpha=0.2, color="green")
    
    if best_max_depth:
        plt.axvline(x=best_max_depth, color='red', linestyle='--', 
                   label=f'Best depth = {best_max_depth}')
    
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main_pipeline(data_path='data.csv', target_col='target'):
    print("=" * 50 + "" + "=" * 50)
    
    # 0. Загрузка данных
    df = load_data(data_path)
    
    # 1. EDA
    print("\n1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
    eda_info = perform_eda(df, target_col)
    
    # 2. Предобработка
    print("\n2. ПРЕДОБРАБОТКА ДАННЫХ")
    X, y, feature_names = preprocess_data(df, target_col)
    
    # 3. Разделение данных
    print("\n3. РАЗДЕЛЕНИЕ ДАННЫХ")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(X, y)
    
    # 4. Масштабирование
    print("\n4. МАСШТАБИРОВАНИЕ ПРИЗНАКОВ")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test, scaler_type='standard'
    )
    
    # 5. Балансировка
    print("\n5. БАЛАНСИРОВКА КЛАССОВ")
    X_train_bal, y_train_bal = balance_classes(X_train_scaled, y_train, balance_method='smote')
    
    # 6. Обучение модели
    print("\n6. ОБУЧЕНИЕ МОДЕЛИ")
    best_model, grid_search, val_curve_data = train_decision_tree(
        X_train_bal, y_train_bal, X_val_scaled, y_val
    )
    
    # 7. Визуализация валидационной кривой
    print("\n7. ВАЛИДАЦИОННАЯ КРИВАЯ")
    best_max_depth = grid_search.best_params_.get('max_depth')
    fig = plot_validation_curve(val_curve_data, best_max_depth)
    
    
    # 8. Оценка на валидационной выборке
    print("\n8. ОЦЕНКА НА ВАЛИДАЦИОННОЙ ВЫБОРКЕ")
    val_metrics = evaluate_model(best_model, X_val_scaled, y_val, X_train_bal, y_train_bal)
    
    # 9. оОценка на тестовой выборке
    print("\n9. ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    X_test_scaled = scaler.transform(X_test)  # На всякий случай повторяем transform
    test_metrics = evaluate_model(best_model, X_test_scaled, y_test, model_name="Decision Tree (Test)")
    
    pipeline_output = {
        'data': {'X': X, 'y': y, 'feature_names': feature_names},
        'split_data': {'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                      'y_train': y_train, 'y_val': y_val, 'y_test': y_test},
        'scaled_data': {'X_train_scaled': X_train_scaled, 'X_val_scaled': X_val_scaled,
                       'X_test_scaled': X_test_scaled, 'scaler': scaler},
        'balanced_data': {'X_train_bal': X_train_bal, 'y_train_bal': y_train_bal},
        'model': best_model,
        'grid_search': grid_search,
        'val_curve': {'data': val_curve_data, 'plot': fig},
        'metrics': {'validation': val_metrics, 'test': test_metrics}
    }
    
    return pipeline_output

# ========== Запуск конвейера ==========
if __name__ == "__main__":
    results = main_pipeline(data_path='data.csv', target_col='target')
    print("\nФинальные результаты:")
    print(f"Лучшая модель: {results['model']}")
    print(f"Test F1-score: {results['metrics']['test']['f1']:.4f}")
    print(f"Test ROC-AUC: {results['metrics']['test'].get('roc_auc', 'N/A')}")