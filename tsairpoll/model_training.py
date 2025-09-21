from pysr import PySRRegressor
#from pygpg.sk import GPGRegressor
#from pyGPGOMEA import GPGOMEARegressor as GPG
import numpy as np
import pandas as pd
#from category_encoders import CountEncoder
#from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNetCV, GammaRegressor, PoissonRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import random
#import smogn
#from ImbalancedLearningRegression import adasyn
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor

from tsairpoll.data_loader import split_data
from tsairpoll.utils import compute_linear_scaling


class CocalOnlyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, -1]


class BasicMedianRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.offset_ = np.median(y)

        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, -1] + self.offset_


class BasicMedianDeltaRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        last_col = X[:, -1]
        delta = y - last_col
        self.offset_ = np.median(delta)

        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, -1] + self.offset_


class DataAugmenterRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, augmenter, pipeline, linear_scaling, log_scale_target):
        self.augmenter = augmenter
        self.pipeline = pipeline
        self.linear_scaling = linear_scaling
        self.log_scale_target = log_scale_target

    def fit(self, X, y):
        X = X.copy().reset_index(drop=True)
        self.augmenter.fit(X, y)
        X_aug, y_aug = self.augmenter.transform(X, y)

        if isinstance(X, pd.DataFrame) and not isinstance(X_aug, pd.DataFrame):
            X_aug = pd.DataFrame(X_aug, columns=X.columns)

        if self.log_scale_target:
            self.pipeline.fit(X_aug, np.log1p(y_aug))
        else:
            self.pipeline.fit(X_aug, y_aug)
        self.slope_ = 1.0
        self.intercept_ = 0.0

        if self.linear_scaling:
            p = self.pipeline.predict(X_aug)
            if self.log_scale_target:
                slope, intercept = compute_linear_scaling(np.log1p(y_aug), p)
            else:
                slope, intercept = compute_linear_scaling(y_aug, p)
            self.slope_ = np.core.umath.clip(slope, -1e+15, 1e+15)
            self.intercept_ = np.core.umath.clip(intercept, -1e+15, 1e+15)

        return self

    def predict(self, X):
        if self.log_scale_target:
            return np.expm1(np.core.umath.clip(self.slope_ * self.pipeline.predict(X), -1e+15, 1e+15) + self.intercept_)
        else:
            return np.core.umath.clip(self.slope_ * self.pipeline.predict(X), -1e+15, 1e+15) + self.intercept_

    def get_params(self, deep=True):
        out = super().get_params(deep=deep)
        out.update({
            'augmenter': self.augmenter,
            'pipeline': self.pipeline,
            'linear_scaling': self.linear_scaling,
            'log_scale_target': self.log_scale_target,
        })
        return out

    def set_params(self, **params):
        super().set_params(**params)

        for key in ['augmenter', 'pipeline', 'linear_scaling', 'log_scale_target']:
            if key in params:
                setattr(self, key, params[key])

        return self


class DummyAugmenter(BaseEstimator):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        return X, y


class RareTargetAugmenter(BaseEstimator):
    def __init__(self, threshold_perc, multiplier, noise_std, y_noise_std, random_state=None):
        self.threshold_perc = threshold_perc
        self.threshold_percentile = int(threshold_perc * 100)
        self.multiplier = multiplier
        self.noise_std = noise_std
        self.y_noise_std = y_noise_std
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.threshold_ = np.percentile(y, self.threshold_percentile)

        # Store numerical columns
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
            self.num_columns_ = X.select_dtypes(include=np.number).columns
        else:
            raise ValueError("RareTargetAugmenter requires X to be a pandas DataFrame.")

        return self

    def transform(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("RareTargetAugmenter requires X to be a pandas DataFrame.")

        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        # Identify rare values
        mask = y > self.threshold_
        X_high = X.loc[mask]
        y_high = y[mask]

        X_reps = []
        y_reps = []

        for _ in range(self.multiplier):
            X_rep = X_high.copy()
            y_rep = y_high.copy()

            if self.noise_std > 0.0:
                noise = rng.normal(loc=0.0, scale=self.noise_std, size=X_high[self.num_columns_].shape)
                X_rep.loc[:, self.num_columns_] += noise

            if self.y_noise_std > 0.0:
                noise = rng.normal(loc=0.0, scale=self.y_noise_std, size=y_high.shape)
                y_rep += noise

            X_reps.append(X_rep)
            y_reps.append(y_rep)

        # Concatenate original and augmented data
        X_aug = pd.concat([X] + X_reps, axis=0).reset_index(drop=True)
        y_aug = np.concatenate([y] + y_reps)

        # Shuffle the final result
        indices = rng.permutation(len(y_aug))
        X_aug = X_aug.iloc[indices].reset_index(drop=True)
        y_aug = y_aug[indices]

        if self.y_noise_std > 0.0:
            y_aug[y_aug < 0.0] = 0.0

        return X_aug, y_aug


# class SmognAugmenter(BaseEstimator):
#     def __init__(self, smogn_params=None, random_state=None):
#         """
#         Initializes the SmognAugmenter.

#         Parameters:
#         - y_col: Name of the target column.
#         - smogn_params: Dictionary of parameters to pass to smogn.smoter.
#         """
#         self.random_state = random_state
#         self.y_col = 'target'
#         self.smogn_params = smogn_params if smogn_params is not None else {}

#     def fit(self, X, y):
#         return self

#     def transform(self, X, y):
#         rng = np.random.default_rng(self.random_state)

#         # Combine X and y into a single DataFrame
#         df = X.copy()

#         df[self.y_col] = y

#         # Apply SMOGN
#         df_smogn = smogn.smoter(data=df.reset_index(drop=True), y=self.y_col, **self.smogn_params)

#         # Separate features and target
#         y_aug = df_smogn[self.y_col].values
#         X_aug = df_smogn.drop(columns=[self.y_col]).reset_index(drop=True)

#         # Shuffle the final result
#         indices = rng.permutation(len(y_aug))
#         X_aug = X_aug.iloc[indices].reset_index(drop=True)
#         y_aug = y_aug[indices]

#         return X_aug, y_aug


# class AdasynAugmenter(BaseEstimator):
#     def __init__(self, adasyn_params=None, random_state=None):
#         """
#         Initializes the AdasynAugmenter.

#         Parameters:
#         - y_col: Name of the target column.
#         - adasyn_params: Dictionary of parameters to pass to adasyn.
#         """
#         self.random_state = random_state
#         self.y_col = 'target'
#         self.adasyn_params = adasyn_params if adasyn_params is not None else {}

#     def fit(self, X, y):
#         return self

#     def transform(self, X, y):
#         rng = np.random.default_rng(self.random_state)

#         # Combine X and y into a single DataFrame
#         df = X.copy()

#         df[self.y_col] = y

#         # Apply ADASYN
#         df_adasyn = adasyn(data=df.reset_index(drop=True), y=self.y_col, **self.adasyn_params)

#         # Separate features and target
#         y_aug = df_adasyn[self.y_col].values
#         X_aug = df_adasyn.drop(columns=[self.y_col]).reset_index(drop=True)

#         # Shuffle the final result
#         indices = rng.permutation(len(y_aug))
#         X_aug = X_aug.iloc[indices].reset_index(drop=True)
#         y_aug = y_aug[indices]

#         return X_aug, y_aug


def preprocess_and_train(df, timestamps, selected_features, encoding_type, scaling_strategy, augmentation_mode, model_name, test_size, n_iter, cv, seed, linear_scaling, log_scale_target, n_train_records, verbose=False):
    if n_train_records < 0:
        raise ValueError('n_train_records must be >= 0')
    if not isinstance(n_train_records, int):
        raise ValueError('n_train_records must be an integer')

    if model_name in ('cocal_only', 'basic_median_delta'):
        encoding_type = 'onehot'
        scaling_strategy = 'none'
        augmentation_mode = 'none'
        log_scale_target = 0
        n_train_records = 0

    random.seed(seed)
    np.random.seed(seed)

    target = df.columns[-1]
    if len(selected_features) == 0:
        df = df.copy()
    else:
        df = df[selected_features + [target]].copy()

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    numerical_cols.remove(target)

    # Split data
    train_df, train_timestamp, test_df, test_timestamp = split_data(df, timestamps, test_size=test_size, random_state=seed, sort=False)

    X_train = train_df.iloc[:,:-1]
    y_train = train_df[target].to_numpy()

    if n_train_records > 0:
        X_train = X_train.iloc[:n_train_records,:]
        y_train = y_train[:n_train_records]

        if isinstance(train_timestamp, pd.Series):
            train_timestamp = train_timestamp.iloc[:n_train_records]
        else:
            train_timestamp = train_timestamp.iloc[:n_train_records, :]

    X_test = test_df.iloc[:, :-1]
    y_test = test_df[target].to_numpy()

    #y_train = np.where(y_train == 0.0, 0.00000001, y_train)

    if encoding_type == "onehot":
        cat_transformer = OneHotEncoder(handle_unknown='ignore')
    #elif encoding_type == "frequency":
    #    cat_transformer = CountEncoder(drop_invariant=True, normalize=True, handle_unknown=0, handle_missing='error')
    else:
        raise ValueError("Invalid encoding type. Choose 'onehot' or 'frequency'.")

    if scaling_strategy == "standard":
        num_transformer = StandardScaler()
    elif scaling_strategy == "minmax":
        num_transformer = MinMaxScaler()
    elif scaling_strategy == "robust":
        num_transformer = RobustScaler()
    elif scaling_strategy == "none":
        num_transformer = "passthrough"
    else:
        raise ValueError("Invalid scaling strategy. Choose 'standard', 'minmax', 'robust', or 'none'.")

    preprocessor = ColumnTransformer([
        ('cat', cat_transformer, categorical_cols),
        ('num', num_transformer, numerical_cols)
    ])

    if augmentation_mode.startswith('simple'):
        simple_aug_params = augmentation_mode.split('_')
        augmenter = RareTargetAugmenter(
            threshold_perc=float(simple_aug_params[1]),
            multiplier=int(simple_aug_params[2]),
            noise_std=float(simple_aug_params[3]),
            y_noise_std=float(simple_aug_params[4]),
            random_state=seed
        )
    # elif augmentation_mode.startswith('smogn'):
    #     smogn_aug_params = augmentation_mode.split('_')
    #     smogn_params = {
    #         'rel_thres': float(smogn_aug_params[1]),
    #         'k': int(smogn_aug_params[2]),
    #         'pert': float(smogn_aug_params[3]),
    #         'samp_method': str(smogn_aug_params[4]), # "balance" or "extreme"
    #         'rel_xtrm_type': str(smogn_aug_params[5]), # "high", "low", or "both"
    #         'rel_method': 'auto',
    #         'seed': seed,
    #         'verbose': False
    #     }
    #     augmenter = SmognAugmenter(smogn_params=smogn_params, random_state=seed)
    # elif augmentation_mode.startswith('adasyn'):
    #     adasyn_aug_params = augmentation_mode.split('_')
    #     adasyn_params = {
    #         'rel_thres': float(adasyn_aug_params[1]),
    #         'k': int(adasyn_aug_params[2]),
    #         'samp_method': str(adasyn_aug_params[3]), # "balance" or "extreme"
    #         'rel_xtrm_type': str(adasyn_aug_params[4]), # "high", "low", or "both"
    #         'rel_method': 'auto',
    #         #'seed': seed,
    #         #'verbose': False
    #     }
    #     augmenter = AdasynAugmenter(adasyn_params=adasyn_params, random_state=seed)
    elif augmentation_mode == "none":
        augmenter = DummyAugmenter(random_state=seed)
    else:
        raise ValueError("Invalid augmentation mode. Choose 'simple', 'smogn', or 'none'.")

    # Model selection
    models = {
        "cocal_only": CocalOnlyRegressor(),
        "basic_median": BasicMedianRegressor(),
        "basic_median_delta": BasicMedianDeltaRegressor(),
        "linear": LinearRegression(),
        "gamma": GammaRegressor(),
        "poisson": PoissonRegressor(),
        "sgd": SGDRegressor(random_state=seed),
        "elasticnet": ElasticNetCV(random_state=seed),
        "random_forest": RandomForestRegressor(random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
        #"lightgbm": LGBMRegressor(random_state=seed),
        "bagging": BaggingRegressor(random_state=seed),
        "adaboost": AdaBoostRegressor(random_state=seed),
        "knn": KNeighborsRegressor(),
        "mlp": MLPRegressor(random_state=seed),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
        "svr": SVR(),
        # "GP-GOMEA": GPGRegressor(
        #     t=-1,  # time limit,
        #     g=-1,  # generations
        #     e=300000,  # fitness evaluations
        #     d=5,  # maximum tree depth
        #     finetune_max_evals=300,  # 10,000 evaluations limit for fine-tuning
        #     pop=256,
        #     bs=256,
        #     tour=4,
        #     fset='+,-,*,/',
        #     ff='ac',
        #     finetune=True,  # whether to fine-tune the coefficients after the search
        #     disable_ims=True,
        #     no_univ_exc_leaves_fos=False,
        #     rci=0.0,
        #     cmp=0.0,
        #     nolink=False,
        #     feat_sel=False,
        #     no_large_fos=True,
        #     random_state=seed,  # for reproducibility
        #     verbose=True,  # print progress
        # ),
        # "GP-GOMEA": GPG(
        #     time=-1,
        #     generations=1000,
        #     evaluations=-1,
        #     popsize=1000,
        #     functions='+_*_-_p/',
        #     tournament=5,
        #     prob='symbreg', multiobj=False, linearscaling=False,
        #     erc=True, classweights=False,
        #     gomea=True, # gomfos='',
        #     #subcross=0.5, submut=0.5, reproduction=0.0,
        #     #sblibtype=False, sbrdo=0.0, sbagx=0.0,
        #     unifdepthvar=True, elitism=1,
        #     ims=False, syntuniqinit=1000,
        #     initmaxtreeheight=2, inittype=False,
        #     maxtreeheight=10, maxsize=40,
        #     validation=False,
        #     coeffmut='0.5_0.5_0.5_10', # False
        #     #gomcoeffmutstrat=False,
        #     batchsize=False,
        #     seed=seed, parallel=1, caching=False,
        #     silent=True, logtofile=False
        # ),
        # "symbolic_regression": PySRRegressor(
        #     random_state=seed,
        #     parallelism='serial',
        #     deterministic=True,
        #     progress=False,
        #     verbosity=0
        # ),
        "symbolic_regression": PySRRegressor(
            niterations=1000,
            maxdepth=10,#9,
            maxsize=30,#25,
            binary_operators=["+", "-", "*", "/", '^'],
            unary_operators=['square', 'abs', 'neg'],
            constraints={'^': (1, 1), 'square': 10},
            complexity_of_operators={'+': 1, '-': 1, '*': 1, '/': 1, '^': 2, 'abs': 1, 'neg': 1, 'square': 2},
            populations=10,
            population_size=1000,#100,
            model_selection="accuracy",
            tournament_selection_n=4,
            crossover_probability=0.3,
            ncycles_per_iteration=10,
            topn=1,
            parsimony=0.0,
            random_state=seed,
            verbosity=0,
            deterministic=True,
            parallelism="serial",
            progress=False,
        ),
    }

    if model_name not in models:
        raise ValueError("Invalid model name")

    model = models[model_name]

    # Hyperparameter grids
    param_grids_0 = {
        "cocal_only": {},
        "basic_median": {},
        "basic_median_delta": {},
        "linear": {},
        "gamma": {
            "regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "regressor__solver": ["lbfgs", "newton-cholesky"],
            "regressor__max_iter": [200, 400, 600, 100]
        },
        "poisson": {
            "regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "regressor__solver": ["lbfgs", "newton-cholesky"],
            "regressor__max_iter": [200, 400, 600, 100]
        },
        "sgd": {
            "regressor__loss": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            "regressor__penalty": ["l2", "l1", "elasticnet"],
            "regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "regressor__l1_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "regressor__learning_rate": ['constant', 'optimal', 'invscaling', 'adaptive'],
        },
        "elasticnet": {
            "regressor__l1_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "regressor__n_alphas": [5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000],
            "regressor__selection": ['cyclic', 'random'],
        },
        "random_forest": {
            "regressor__n_estimators": [10, 20, 50, 100, 150, 200],
            "regressor__oob_score": [True, False],
            "regressor__max_depth": [3, 6, 9, 12, 15],
            "regressor__max_features": ['sqrt', 'log2', None],
            "regressor__min_samples_split": [2, 4, 6, 8],
            "regressor__min_samples_leaf": [0.001, 0.01, 0.1, 1, 2, 4],
            "regressor__criterion": ['friedman_mse', 'squared_error', 'absolute_error', 'poisson'],
        },
        "gradient_boosting": {
            "regressor__loss": ['squared_error', 'absolute_error', 'huber', 'quantile'],
            "regressor__learning_rate": [1e-4, 1e-3, 1e-2, 1e-1],
            "regressor__n_estimators": [10, 20, 50, 100, 150, 200],
            "regressor__criterion": ['friedman_mse', 'squared_error'],
            "regressor__max_features": ['sqrt', 'log2'],
            "regressor__max_depth": [3, 6, 9, 12, 15],
            "regressor__min_samples_split": [2, 4, 6, 8],
            "regressor__min_samples_leaf": [0.001, 0.01, 0.1, 1, 2, 4],
            "regressor__validation_fraction": [0.1, 0.2, 0.3],
            "regressor__n_iter_no_change": [10, 20, 30],
        },
        # "lightgbm": {
        #     "regressor__boosting_type": ["gbdt", "dart", "rf"],
        #     "regressor__max_depth": [3, 6, 9, 12, 15],
        #     "regressor__learning_rate": [1e-4, 1e-3, 1e-2, 1e-1],
        #     "regressor__n_estimators": [10, 20, 50, 100, 150, 200],
        #     "regressor__num_leaves": [10, 20, 30, 50, 100],
        #     "regressor__reg_alpha": [0.0, 0.1, 0.3, 0.7],
        #     "regressor__reg_lambda": [0.0, 0.1, 0.3, 0.7],
        #     "regressor__importance_type": ["split", "gain"],
        #     "regressor__min_data_in_leaf": [20, 50, 100, 200, 300, 500],
        #     "regressor__max_bin": [10, 20, 40, 60, 80, 255],
        # },
        "bagging": {
            "regressor__n_estimators": [10, 20, 50, 100, 150, 200],
            "regressor__oob_score": [True, False],
        },
        "adaboost": {
            "regressor__n_estimators": [10, 20, 50, 100, 150, 200],
            "regressor__learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "regressor__loss": ["linear", "square", "exponential"],
        },
        "knn": {
            "regressor__n_neighbors": [3, 5, 7],
            "regressor__weights": ['uniform', 'distance'],
            "regressor__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        },
        "mlp": {
            "regressor__hidden_layer_sizes": [(50,), (20,), (50,50), (20,20), (50,50,50), (20,20,20)],
            "regressor__activation": ['tanh', 'relu', 'logistic'],
            "regressor__solver": ['adam'],
            "regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "regressor__batch_size": [16, 32, 64, 'auto'],
            "regressor__learning_rate": ['constant', 'invscaling', 'adaptive'],
            "regressor__max_iter": [200, 400, 600],
            "regressor__validation_fraction": [0.1, 0.2, 0.3],
            "regressor__n_iter_no_change": [10, 20, 30],
            "regressor__early_stopping": [True, False],
        },
        "decision_tree": {
            "regressor__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "regressor__splitter": ["best", "random"],
            "regressor__max_depth": [3, 6, 9, 12, 15],
            "regressor__max_features": ['sqrt', 'log2'],
        },
        "svr": {
            "regressor__kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            "regressor__C": [0.01, 0.1, 0.5, 1, 10, 100, 1000],
            "regressor__gamma": [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0],
            "regressor__degree": [1, 2, 3],
            "regressor__epsilon": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 0.2, 0.3, 0.5],
            "regressor__coef0": [0.0, 0.1, 0.2, 0.3],
            "regressor__shrinking": [True, False],
            "regressor__max_iter": [200, 400, 600, -1]
        },
        "symbolic_regression": {},
        # "symbolic_regression": {
        #     "regressor__maxsize": [30],
        #     "regressor__maxdepth": [15],
        #     "regressor__populations": [1],
        #     "regressor__population_size": [100],
        #     "regressor__niterations": [100],
        #     "regressor__binary_operators": [['+', '-', '*', '/'], ['+', '-', '*', '/', '^'], ['+', '-', '*', '/', 'max', 'min'], ['+', '-', '*', '/', '^', 'max', 'min']],
        #     "regressor__unary_operators": [['neg', 'square'], ['square', 'sqrt'], ['square', 'log'], ['neg', 'square', 'sqrt'], ['neg', 'square', 'sqrt', 'log']],
        #     "regressor__parsimony": [0.0],
        #     "regressor__complexity_of_operators": [{'+': 1, '-': 1, '*': 1, '/': 1, '^': 2, 'max': 1, 'min': 1, 'neg': 1, 'square': 1, 'sqrt': 2, 'log': 3}],
        #     "regressor__constraints": [{'^': (-1, 5), 'sqrt': 5, 'log': 5, 'square': 10}],
        #     "regressor__ncycles_per_iteration": [5],
        #     "regressor__tournament_selection_n": [4],
        #     "regressor__batching": [False],
        #     "regressor__model_selection": ['accuracy'],
        #     "regressor__topn": [1],
        #     "regressor__fast_cycle": [True],
        #     "regressor__crossover_probability": [0.5],
        # },
        #"GP-GOMEA": {
        #    "regressor__no_univ_exc_leaves_fos": [True, False],
        #    "regressor__cmp": [0.0, 0.1],
        #    "regressor__rci": [0.0, 0.1],
        #}
        #"GP-GOMEA": {},
    }

    param_grids = {k: {k1.replace('regressor__', 'pipeline__regressor__'): param_grids_0[k][k1] for k1 in param_grids_0[k]} for k in param_grids_0}

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Create regressor with augmentation
    augmented_regressor = DataAugmenterRegressor(augmenter=augmenter, pipeline=pipeline, linear_scaling=linear_scaling, log_scale_target=log_scale_target)

    # Hyperparameter tuning
    search = RandomizedSearchCV(augmented_regressor, param_grids[model_name], n_iter=n_iter, cv=cv, n_jobs=1,
                                scoring='neg_root_mean_squared_error', random_state=seed, refit=True, verbose=verbose)
    search.fit(X_train, y_train)

    return search, X_train, X_test, y_train, y_test, train_timestamp, test_timestamp
