from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error


def evaluate_model(pipeline, X_train, X_test, y_train, y_test, metric="r2"):
    """
    Evaluate the trained pipeline on train and test data using the specified metric.

    Parameters:
    - pipeline: The trained scikit-learn pipeline (RandomizedSearchCV output)
    - X_train, X_test: Feature matrices for training and testing
    - y_train, y_test: Target values for training and testing
    - metric: String specifying the evaluation metric ("r2", "mae", "mse")
    """
    metrics = {
        "r2": r2_score,
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error
    }

    if metric not in metrics:
        raise ValueError(f"Unsupported metric: {metric}. Choose from {list(metrics.keys())}")

    scoring_function = metrics[metric]

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    cocal_train = X_train.iloc[:, -1].to_numpy().flatten()
    cocal_test = X_test.iloc[:, -1].to_numpy().flatten()

    train_score_sum_cocal = scoring_function(y_train + cocal_train, y_train_pred + cocal_train)
    test_score_sum_cocal = scoring_function(y_test + cocal_test, y_test_pred + cocal_test)

    train_score = scoring_function(y_train, y_train_pred)
    test_score = scoring_function(y_test, y_test_pred)

    return {"train_score": train_score, "test_score": test_score, "train_score_sum_cocal": train_score_sum_cocal, "test_score_sum_cocal": test_score_sum_cocal, "train_pred": y_train_pred, "test_pred": y_test_pred}
