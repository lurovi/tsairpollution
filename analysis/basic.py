import random
import warnings
import zlib

import fastplot
import seaborn as sns

import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import random


from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from tsairpoll.data_loader import create_dir_path_results, load_data, split_data
import statistics
import json
from prettytable import PrettyTable
import os

from tsairpoll.model_evaluation import evaluate_model
from tsairpoll.utils import load_pkl, only_first_char_upper
from tsairpoll.stat_test import perform_mannwhitneyu_holm_bonferroni


# ======================================================================================
# SIMPLE ANALYSIS
# ======================================================================================


def print_formula_sr(path, dataset, features, selected_features, encoding, scaling, augmentation, test_size, n_iter, cv, linear_scaling, log_scale_target, n_train_records, seed_index):
    df = load_data(f'../data/formatted/{dataset}.csv')
    timestamps = load_data(f'../data/formatted/{dataset}_timestamps.csv')

    with open('../random_seeds.txt', 'r') as f:
        # THE ACTUAL SEED TO BE USED IS LOCATED AT POSITION seed_index - 1 SINCE seed_index IS AN INDEX THAT STARTS FROM 1
        all_actual_seeds = [int(curr_actual_seed_as_str) for curr_actual_seed_as_str in f.readlines()]
    seed = all_actual_seeds[seed_index - 1]

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
    train_df, train_timestamp, test_df, test_timestamp = split_data(df, timestamps, test_size=test_size,
                                                                    random_state=seed, sort=False)

    X_train = train_df.iloc[:, :-1]
    y_train = train_df[target].to_numpy()

    print(X_train.columns)

    if encoding == "onehot":
        cat_transformer = OneHotEncoder(handle_unknown='ignore')
    elif encoding == "frequency":
        cat_transformer = CountEncoder(drop_invariant=True, normalize=True, handle_unknown=0, handle_missing='error')
    else:
        raise ValueError("Invalid encoding type. Choose 'onehot' or 'frequency'.")

    if scaling == "standard":
        num_transformer = StandardScaler()
    elif scaling == "minmax":
        num_transformer = MinMaxScaler()
    elif scaling == "robust":
        num_transformer = RobustScaler()
    elif scaling == "none":
        num_transformer = "passthrough"
    else:
        raise ValueError("Invalid scaling strategy. Choose 'standard', 'minmax', 'robust', or 'none'.")

    # "season,time_of_day,dew_point,NUM_MEASUREMENTS,BM_P_MEAN,GPS_KMH_MEAN,GPS_DIR_MEAN,GPS_ALT_MEAN,PM10_MEAN"

    preprocessor = ColumnTransformer([
        ('cat', cat_transformer, categorical_cols),
        ('num', num_transformer, numerical_cols)
    ])

    preprocessor.fit(X_train, y_train)
    X_transformed = preprocessor.transform(X_train)
    print(preprocessor.named_transformers_['cat'].get_feature_names_out())

    s = create_dir_path_results(
        base_path=path,
        dataset=dataset,
        features=features,
        encoding=encoding,
        scaling=scaling,
        augmentation=augmentation,
        model='symbolic_regression',
        test_size=test_size,
        n_iter=n_iter,
        cv=cv,
        linear_scaling=linear_scaling,
        log_scale_target=log_scale_target,
        n_train_records=n_train_records,
    )

    search = load_pkl(os.path.join(s, f'search{seed_index}.pkl'))
    print(search.best_estimator_.pipeline['regressor'].latex())

    with open(os.path.join(s, f'result{seed_index}.json'), 'r') as f:
        res = json.load(f)

    print("Volontari Test MAE ", res['volontari']['mae']["test_score"])
    print("Carpineto Test MAE ", res['carpineto']['mae']["test_score"])
    print("Sincrotrone Test MAE ", res['sincrotrone']['mae']["test_score"])



def print_basic_scores_with_cap(y_cap, var_cap_id, var_cap_value, path, dataset, test_dataset, features, encoding, scaling, augmentation, models, test_size, n_iter, cv, linear_scaling, log_scale_target, n_train_records, seed_indexes):
    if test_dataset is not None:
        test_real = load_data('../' + test_dataset)
        test_timestamps_real = load_data('../' + test_dataset.replace('.csv', '_timestamps.csv'))
    metrics = ["r2", "rmse", "mae"]
    tables = dict()
    for metric in metrics:
        table = PrettyTable()
        table.field_names = ["Model", "Q1 (Train)", "Median (Train)", "Q3 (Train)", "Q1 (Test)", "Median (Test)", "Q3 (Test)"]
        tables[metric] = table

    for model in models:
        s = create_dir_path_results(
            base_path=path,
            dataset=dataset,
            features=features,
            encoding=encoding,
            scaling=scaling,
            augmentation=augmentation,
            model=model,
            test_size=test_size,
            n_iter=n_iter,
            cv=cv,
            linear_scaling=linear_scaling,
            log_scale_target=log_scale_target,
            n_train_records=n_train_records,
        )
        train_scores = {metric: [] for metric in metrics}
        test_scores = {metric: [] for metric in metrics}

        for seed_index in seed_indexes:
            with open(os.path.join(s, f'result{seed_index}.json'), 'r') as f:
                res = json.load(f)
            search = load_pkl(os.path.join(s, f'search{seed_index}.pkl'))
            data_path = res['data']
            timestamps_path = res['timestamps']
            selected_features = res['selected_features']
            actual_seed = res['seed']
            df = load_data('../' + data_path)
            df_timestamps = load_data('../' + timestamps_path)
            random.seed(actual_seed)
            np.random.seed(actual_seed)

            target = df.columns[-1]
            if len(selected_features) == 0:
                df = df.copy()
            else:
                df = df[selected_features + [target]].copy()

            # Split data
            train_df, train_timestamp, test_df, test_timestamp = split_data(df, df_timestamps, test_size=test_size, random_state=actual_seed, sort=False)

            X_train = train_df.iloc[:, :-1]
            y_train = train_df[target].to_numpy()

            X_test = test_df.iloc[:, :-1]
            y_test = test_df[target].to_numpy()

            if n_train_records > 0:
                X_train = X_train.iloc[:n_train_records, :]
                y_train = y_train[:n_train_records]

                if isinstance(train_timestamp, pd.Series):
                    train_timestamp = train_timestamp.iloc[:n_train_records]
                else:
                    train_timestamp = train_timestamp.iloc[:n_train_records, :]

            y_train = np.where(y_train == 0.0, 0.00000001, y_train)

            if y_cap is not None and y_cap > 0:
                mask = y_train <= y_cap
                X_train = X_train[mask]
                y_train = y_train[mask]

                mask = y_test <= y_cap
                X_test = X_test[mask]
                y_test = y_test[mask]

            if var_cap_id is not None and var_cap_value is not None and var_cap_value > 0 and var_cap_id >= 0:
                mask = X_train.iloc[:, var_cap_id] <= var_cap_value
                X_train = X_train[mask]
                y_train = y_train[mask]

                mask = X_test.iloc[:, var_cap_id] <= var_cap_value
                X_test = X_test[mask]
                y_test = y_test[mask]

            for metric in metrics:
                evaluation = evaluate_model(search, X_train, X_test, y_train, y_test, metric)

                train_scores[metric].append(evaluation['train_score'])
                test_scores[metric].append(evaluation['test_score'])

        for metric in metrics:
            train_med = statistics.median(train_scores[metric])
            train_q1 = float(np.percentile(train_scores[metric], 25))
            train_q3 = float(np.percentile(train_scores[metric], 75))

            test_med = statistics.median(test_scores[metric])
            test_q1 = float(np.percentile(test_scores[metric], 25))
            test_q3 = float(np.percentile(test_scores[metric], 75))

            tables[metric].add_row([model, train_q1, train_med, train_q3, test_q1, test_med, test_q3])

    for metric in metrics:
        print("==========" + metric + "==========")
        print()
        print(tables[metric])
        print()
        print()


def print_basic_scores(metric, path, dataset, features, encoding, scaling, augmentation, models, test_size, n_iter, cv, linear_scaling, log_scale_target, n_train_records, seed_indexes):
    table = PrettyTable()
    table.field_names = ["Model", "Q1 (Train)", "Median (Train)", "Q3 (Train)", "Q1 (Test)", "Median (Test)", "Q3 (Test)"]

    table_sum_cocal = PrettyTable()
    table_sum_cocal.field_names = ["Model", "Q1 (Train)", "Median (Train)", "Q3 (Train)", "Q1 (Test)", "Median (Test)", "Q3 (Test)"]

    for model in models:
        s = create_dir_path_results(
            base_path=path,
            dataset=dataset,
            features=features,
            encoding=encoding,
            scaling=scaling,
            augmentation=augmentation,
            model=model,
            test_size=test_size,
            n_iter=n_iter,
            cv=cv,
            linear_scaling=linear_scaling,
            log_scale_target=log_scale_target,
            n_train_records=n_train_records,
        )
        train_scores = []
        test_scores = []

        train_sum_cocal_scores = []
        test_sum_cocal_scores = []

        for seed_index in seed_indexes:
            with open(os.path.join(s, f'result{seed_index}.json'), 'r') as f:
                res = json.load(f)
            train_scores.append(res[metric]['train_score'])
            test_scores.append(res[metric]['test_score'])
            train_sum_cocal_scores.append(res[metric]['train_score_sum_cocal'])
            test_sum_cocal_scores.append(res[metric]['test_score_sum_cocal'])

        train_med = statistics.median(train_scores)
        train_q1 = float(np.percentile(train_scores, 25))
        train_q3 = float(np.percentile(train_scores, 75))

        test_med = statistics.median(test_scores)
        test_q1 = float(np.percentile(test_scores, 25))
        test_q3 = float(np.percentile(test_scores, 75))

        table.add_row([model, train_q1, train_med, train_q3, test_q1, test_med, test_q3])

        train_sum_cocal_med = statistics.median(train_sum_cocal_scores)
        train_sum_cocal_q1 = float(np.percentile(train_sum_cocal_scores, 25))
        train_sum_cocal_q3 = float(np.percentile(train_sum_cocal_scores, 75))

        test_sum_cocal_med = statistics.median(test_sum_cocal_scores)
        test_sum_cocal_q1 = float(np.percentile(test_sum_cocal_scores, 25))
        test_sum_cocal_q3 = float(np.percentile(test_sum_cocal_scores, 75))

        table_sum_cocal.add_row([model, train_sum_cocal_q1, train_sum_cocal_med, train_sum_cocal_q3, test_sum_cocal_q1, test_sum_cocal_med, test_sum_cocal_q3])

    print("==========" + metric + "==========")
    print()
    if path.endswith('delta') or path.endswith('delta/'):
        print('with delta')
        print(table_sum_cocal)
    else:
        print(table)
    print()
    print()

# ======================================================================================
# PLOTS
# ======================================================================================


def create_lineplot_single_repetition(path, dataset, features, selected_features, encoding, scaling, augmentation, model, test_size, n_iter, cv, linear_scaling, log_scale_target, n_train_records, seed_index):
    df = load_data(f'../data/formatted/{dataset}.csv')
    timestamps = load_data(f'../data/formatted/{dataset}_timestamps.csv')

    with open('../random_seeds.txt', 'r') as f:
        # THE ACTUAL SEED TO BE USED IS LOCATED AT POSITION seed_index - 1 SINCE seed_index IS AN INDEX THAT STARTS FROM 1
        all_actual_seeds = [int(curr_actual_seed_as_str) for curr_actual_seed_as_str in f.readlines()]
    seed = all_actual_seeds[seed_index - 1]

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
    train_df, train_timestamp, test_df, test_timestamp = split_data(df, timestamps, test_size=test_size,
                                                                    random_state=seed, sort=False)

    X_train = train_df.iloc[:, :-1]
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

    y_train = np.where(y_train == 0.0, 0.00000001, y_train)

    s = create_dir_path_results(
        base_path=path,
        dataset=dataset,
        features=features,
        encoding=encoding,
        scaling=scaling,
        augmentation=augmentation,
        model=model,
        test_size=test_size,
        n_iter=n_iter,
        cv=cv,
        linear_scaling=linear_scaling,
        log_scale_target=log_scale_target,
        n_train_records=n_train_records,
    )

    search = load_pkl(os.path.join(s, f'search{seed_index}.pkl'))

    s = create_dir_path_results(
        base_path=path,
        dataset=dataset,
        features=features,
        encoding=encoding,
        scaling=scaling,
        augmentation=augmentation,
        model=model,
        test_size=test_size,
        n_iter=n_iter,
        cv=cv,
        linear_scaling=linear_scaling,
        log_scale_target=log_scale_target,
        n_train_records=n_train_records,
    )

    search = load_pkl(os.path.join(s, f'search{seed_index}.pkl'))


def my_callback_function_that_actually_draws_boxplot(plt, data, marked_models, dataset_split_palette, x_label="Model", y_label="MAE", title=None):
    figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    # Grid and ticks
    ax.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # Draw boxplot
    sns.boxplot(data=data, x="Model", y="MAE", hue="Dataset", palette=dataset_split_palette, legend=True,
                log_scale=None, fliersize=0.0, showfliers=False, ax=ax)

    # Annotate stars for marked models
    ymin, _ = ax.get_ylim()
    star_y = ymin + (0.03 * (ax.get_ylim()[1] - ymin))  # a bit below ymin

    xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
    for xpos, model_label in enumerate(xtick_labels):
        for key in marked_models:
            # Check if this label corresponds to this key
            if model_label == key:
                if marked_models[key]:
                    ax.text(
                        xpos, star_y, r"\textbf{*}",
                        ha='center', va='top', fontsize=14, color='black', clip_on=False
                    )
                break

def export_mae_boxplot(path, zone, dataset, features, encoding, scaling, augmentation, models, test_size, n_iter, cv,
                        linear_scaling, log_scale_target, n_train_records, seed_indexes,
                        dataset_split_palette, dpi, PLOT_ARGS):
    all_data = []

    model_values = {}

    for model in models:
        if model in ('cocal_only', 'basic_median_delta'):
            model_path = create_dir_path_results(
                base_path=path,
                dataset=dataset,
                features=features,
                encoding=encoding,
                scaling='none',
                augmentation='none',
                model=model,
                test_size=test_size,
                n_iter=n_iter,
                cv=cv,
                linear_scaling=0,
                log_scale_target=0,
                n_train_records=0,
            )
        else:
            model_path = create_dir_path_results(
                base_path=path,
                dataset=dataset,
                features=features,
                encoding=encoding,
                scaling=scaling,
                augmentation=augmentation,
                model=model,
                test_size=test_size,
                n_iter=n_iter,
                cv=cv,
                linear_scaling=linear_scaling,
                log_scale_target=log_scale_target,
                n_train_records=n_train_records,
            )

        temp = []

        for seed_index in seed_indexes:
            with open(os.path.join(model_path, f'result{seed_index}.json'), 'r') as f:
                res = json.load(f)

            train_mae = res[zone]["mae"]["train_score"]
            test_mae = res[zone]["mae"]["test_score"]

            model_string = ' '.join([only_first_char_upper(sss) for sss in (model.replace('_', ' ').capitalize() if len(model) > 3 else model.upper()).replace("Cocal", "COCAL").split(' ')])

            temp.append(test_mae)
            all_data.append({"Model": model_string, "Dataset": "Train", "MAE": train_mae})
            all_data.append({"Model": model_string, "Dataset": "Test", "MAE": test_mae})

        model_values[model] = temp

    holm, mann = perform_mannwhitneyu_holm_bonferroni(model_values, alternative='less')
    marked_models = {}
    for model in models:
        model_string = ' '.join([only_first_char_upper(sss) for sss in (model.replace('_', ' ').capitalize() if len(model) > 3 else model.upper()).replace("Cocal", "COCAL").split(' ')])

        marked_models[model_string] = False
        if holm[model]:
            marked_models[model_string] = True
        else:
            temp = mann[model]
            for key in temp:
                if temp[key]:
                    marked_models[model_string] = True
                    break

    df = pd.DataFrame(all_data)

    plot = fastplot.plot(None, None, mode='callback',
                         callback=lambda plt: my_callback_function_that_actually_draws_boxplot(plt, df, marked_models, dataset_split_palette),
                         style='latex', **PLOT_ARGS)

    plot.savefig('boxplot.pdf', dpi=dpi)
    plot.savefig('boxplot.png', dpi=dpi)


def draw_mae_lineplot(plt, summary, x_labels, model, dataset_split_palette):
    figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    x = np.arange(len(x_labels))

    for label, style in zip(["train", "test"], [(dataset_split_palette["Train"], "Train"), (dataset_split_palette["Test"], "Test")]):
        medians = [entry["median"] for entry in summary[label]]
        q1 = [entry["q1"] for entry in summary[label]]
        q3 = [entry["q3"] for entry in summary[label]]

        ax.plot(x, medians, label=style[1], color=style[0], linestyle='-', linewidth=1.4, markersize=10)
        ax.fill_between(x, q1, q3, alpha=0.3, color=style[0])

    # Grid and ticks
    ax.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)

    ax.set_ylim(1, 7)
    ax.set_yticks([1, 3, 5, 7])

    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Number of Training Records")
    ax.set_ylabel("MAE")
    ax.set_title(f"{model.replace('_', ' ').capitalize()} MAE vs Training Size")
    ax.legend()


def collect_mae_lineplot_data(path, dataset, features, encoding, scaling, augmentation, model, test_size, n_iter, cv,
                              linear_scaling, log_scale_target, n_train_record_list, seed_indexes,
                              dataset_split_palette, dpi, PLOT_ARGS):
    summary = {"train": [], "test": []}
    x_labels = []

    for n_train_records in n_train_record_list:
        train_maes = []
        test_maes = []

        result_path = create_dir_path_results(
            base_path=path,
            dataset=dataset,
            features=features,
            encoding=encoding,
            scaling=scaling,
            augmentation=augmentation,
            model=model,
            test_size=test_size,
            n_iter=n_iter,
            cv=cv,
            linear_scaling=linear_scaling,
            log_scale_target=log_scale_target,
            n_train_records=n_train_records,
        )

        for seed_index in seed_indexes:
            with open(os.path.join(result_path, f'result{seed_index}.json'), 'r') as f:
                res = json.load(f)

            train_maes.append(res["mae"]["train_score"])
            test_maes.append(res["mae"]["test_score"])

        summary["train"].append({
            "median": np.median(train_maes),
            "q1": np.percentile(train_maes, 25),
            "q3": np.percentile(train_maes, 75)
        })

        summary["test"].append({
            "median": np.median(test_maes),
            "q1": np.percentile(test_maes, 25),
            "q3": np.percentile(test_maes, 75)
        })

        x_labels.append("max" if n_train_records == 0 else str(n_train_records))

    # Use fastplot to draw the plot
    plot = fastplot.plot(None, None, mode='callback',
                         callback=lambda plt: draw_mae_lineplot(plt, summary, x_labels, model, dataset_split_palette),
                         style='latex', **PLOT_ARGS)

    plot.savefig(f'mae_{model}_lineplot.pdf', dpi=dpi)
    plot.savefig(f'mae_{model}_lineplot.png', dpi=dpi)


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    preamble = r'''
                        \usepackage{amsmath}
                        \usepackage{libertine}
                        \usepackage{xspace}
                        '''

    PLOT_ARGS = {'rcParams': {'text.latex.preamble': preamble, 'pdf.fonttype': 42, 'ps.fonttype': 42}}

    dataset_split_palette = {'Train': '#625DF2', 'Test': '#D77A1D'}

    #path = '../../../python_remote_data/ts-air-pollution/results/'
    path = '../results/'
    dataset = 'volontari'
    features = "season,time_of_day,dew_point,NUM_MEASUREMENTS,BM_P_MEAN,GPS_KMH_MEAN,GPS_DIR_MEAN,GPS_ALT_MEAN,PM10_MEAN"
    selected_features = features.split(',')
    features = f'features{zlib.adler32(bytes(features, "utf-8"))}'
    encoding = 'onehot'
    scaling = 'robust'
    augmentation = 'none' # 'adasyn_0.90_5_balance_both' # 'smogn_0.90_5_0.04_balance_both' # 'simple_0.90_5_0.04_0.04'
    test_size = 0.3
    n_iter = 40
    cv = 4
    linear_scaling = 0
    log_scale_target = 0
    n_train_records = 0
    seed_indexes = list(range(1, 30 + 1))
    #models = ["cocal_only", "basic_median_delta", "linear", "elasticnet", "decision_tree", "symbolic_regression", "svr", "random_forest", "bagging", "gradient_boosting", "adaboost", "mlp"]
    #models = ["cocal_only", "basic_median_delta", "linear", "elasticnet", "decision_tree", "symbolic_regression"]
    models = ["elasticnet", "svr", "random_forest", "bagging", "gradient_boosting", "adaboost", "mlp"]

    #print_basic_scores_with_cap(None, None, None, path=path, dataset=dataset, test_dataset=None, features=features, encoding=encoding, scaling=scaling, augmentation=augmentation, models=models, test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling, log_scale_target=log_scale_target, n_train_records=n_train_records, seed_indexes=seed_indexes)

    export_mae_boxplot(path=path, zone='sincrotrone', dataset=dataset, features=features, encoding=encoding, scaling=scaling, augmentation=augmentation, models=models, test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling, log_scale_target=log_scale_target, n_train_records=n_train_records, seed_indexes=seed_indexes, dataset_split_palette=dataset_split_palette, dpi=800, PLOT_ARGS=PLOT_ARGS)
    #collect_mae_lineplot_data(path=path, dataset=dataset, features=features, encoding=encoding, scaling=scaling, augmentation=augmentation, model='gradient_boosting', test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling, log_scale_target=log_scale_target, n_train_record_list=[400, 800, 1200, 1600, 2000, 0], seed_indexes=seed_indexes, dataset_split_palette=dataset_split_palette, dpi=500, PLOT_ARGS=PLOT_ARGS)

    #print_formula_sr(path=path, dataset=dataset, features=features, selected_features=selected_features, encoding=encoding, scaling=scaling, augmentation=augmentation, test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling,
    #                 log_scale_target=log_scale_target, n_train_records=n_train_records, seed_index=12)


if __name__ == '__main__':
    main()
