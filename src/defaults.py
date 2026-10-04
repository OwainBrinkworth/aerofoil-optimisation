dryrun = False
replacement = False

param_bounds = {
    "R_LE": (0.005, 0.03),
    "x_PRE": (0.3, 0.6),
    "y_PRE": (-0.02, 0.0),
    "d2y/dx2_PRE": (-0.5, -0.05),
    "th_PRE": (-2, 5),
    "x_SUC": (0.2, 0.5),
    "y_SUC": (0.03, 0.12),
    "d2y/dx2_SUC": (-0.6, -0.1),
    "th_SUC": (-10, -2),
}

params = {
    'experiment': {
        'n_samples': 10,
        'sampling_method': 'LHS',
    },

    'solve': {
        'Angle_of_attack': [6],
        'Reynolds_number': 3e6,
        'Iterations': 200,
    },

    'model': {
        'type': 'random_forest',

        # Random Forest
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_leaf': 1,
        },

        # Gradient Boosting
        'gradient_boosting': {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 3,
        },

        # Linear
        'linear': {},

        # KNN
        'knn': {
            'n_neighbors': 10
        },

        # SVR
        'svr': {
            'kernel': 'rbf',
            'C': 10,
            'epsilon': 0.01,
        },

        # Neural Network
        'mlp': {
            'hidden_layer_sizes': (64, 64),
            'activation': 'relu',
            'max_iter': 500,
        },

        # Gaussian Process
        'gpr': {
            'length_scale': 1.0,
            'n_restarts': 3,
        },

        # XGBoost
        'xgboost': {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
    },

    'test': {
        'interpolation': True,
        'n_samples': 2000,
        'sampling_method': 'LHS',
        'expansion_ratio': 1.2,
    }
}

post = {
    'CL': (0.0, 1.4),
    'error': (0.0, 1.0),
}
