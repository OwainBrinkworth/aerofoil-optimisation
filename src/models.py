import time
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from defaults import params


def train_surrogate_model(results_df, test_size=None, random_state=42):

    df_clean = results_df.dropna(subset=["CL"])
    if len(df_clean) == 0:
        raise ValueError("No valid CL values found!")

    X = df_clean.drop(columns=["CL"])
    y = df_clean["CL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model_type = params['model']['type']

    # =========================
    # Model selection
    # =========================

    # Random Forest
    if model_type == "random_forest":
        cfg = params['model']['random_forest']
        model = RandomForestRegressor(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            min_samples_leaf=cfg['min_samples_leaf'],
            random_state=random_state
        )

    # Gradient Boosting
    elif model_type == "gradient_boosting":
        cfg = params['model']['gradient_boosting']
        model = GradientBoostingRegressor(
            n_estimators=cfg['n_estimators'],
            learning_rate=cfg['learning_rate'],
            max_depth=cfg['max_depth']
        )

    # Linear
    elif model_type == "linear":
        model = LinearRegression()

    # KNN (scaled)
    elif model_type == "knn":
        cfg = params['model']['knn']
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=cfg['n_neighbors']))
        ])

    # SVR (scaled)
    elif model_type == "svr":
        cfg = params['model']['svr']
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(
                kernel=cfg['kernel'],
                C=cfg['C'],
                epsilon=cfg['epsilon']
            ))
        ])

    # Neural Network (scaled)
    elif model_type == "mlp":
        cfg = params['model']['mlp']
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=cfg['hidden_layer_sizes'],
                activation=cfg['activation'],
                max_iter=cfg['max_iter'],
                random_state=random_state
            ))
        ])

    # Gaussian Process
    elif model_type == "gpr":
        cfg = params['model']['gpr']
        kernel = ConstantKernel(1.0) * RBF(length_scale=cfg['length_scale'])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=cfg['n_restarts']
            ))
        ])

    # XGBoost
    elif model_type == "xgboost":
        cfg = params['model']['xgboost']
        model = XGBRegressor(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            learning_rate=cfg['learning_rate'],
            subsample=cfg['subsample'],
            colsample_bytree=cfg['colsample_bytree'],
            random_state=random_state
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # =========================
    # Train
    # =========================
    start_time = time.time()
    model.fit(X_train, y_train)
    time_to_train = time.time() - start_time

    return model, X_test, y_test, time_to_train
