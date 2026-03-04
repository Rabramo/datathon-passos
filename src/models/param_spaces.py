# src/models/param_spaces.py
def get_param_distributions(model_name: str):
    if model_name == "logreg":
        return {
            "model__C": [0.01, 0.1, 1.0, 3.0, 10.0],
            "model__class_weight": [None, "balanced"],
            # não tunar penalty/solver para evitar FutureWarning nas versões novas
        }

    if model_name == "tree":
        return {
            "model__max_depth": [None, 3, 5, 8, 12, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__class_weight": [None, "balanced"],
        }

    if model_name == "rf":
        return {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 6, 10, 16, 24],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__class_weight": [None, "balanced"],
        }

    if model_name == "xgb":
        return {
            "model__n_estimators": [300, 600, 900],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 4, 6, 8],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0],
            "model__reg_alpha": [0.0, 0.1, 0.5],
        }

    if model_name == "cat":
        return {
            "model__iterations": [400, 800, 1200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__depth": [4, 6, 8, 10],
            "model__l2_leaf_reg": [1.0, 3.0, 10.0],
        }

    if model_name == "dummy":
        return {}

    raise ValueError(f"Modelo desconhecido: {model_name}")