import random
import warnings
import zlib

import fastplot
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd
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

import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import contextily as ctx  # for background tiles


# ======================================================================================
# MODEL EVALUATION
# ======================================================================================

def apply_aggregated_model_from_repetitions(path, model, dataset_name, features, selected_features, encoding, scaling, augmentation, test_size, n_iter, cv, linear_scaling, log_scale_target, n_train_records, seed_indexes):
    """
    Given a model type and a dataset, it loads all the models from all the repetitions.
    Each model is applied to the dataset to obtain a prediction vector.
    Then, the predictions are aggregated by median to obtain a final prediction vector.
    Moreover, the prediction vectors from the repetitions are also aggregated by Q1 and Q3,
    to have an idea of the variability of the predictions.
    The methods returns timestamps of the rows of the dataset, the coordinates <lon, lat>,
    the values at the last column of the dataset (PM10_MEAN),
    the real target values (PM10_ARPA), the aggregated prediction vector (median), Q1 vector and Q3 vector.
    """
    # Load all models from all repetitions
    models = []
    if model in ('cocal_only', 'basic_median_delta'):
        s = create_dir_path_results(
            base_path=path,
            dataset='volontari',
            features=features,
            encoding='onehot',
            scaling='none',
            augmentation='none',
            model=model,
            test_size=test_size,
            n_iter=n_iter,
            cv=cv,
            linear_scaling=linear_scaling,
            log_scale_target=0,
            n_train_records=0,
        )
    else:
        s = create_dir_path_results(
            base_path=path,
            dataset='volontari',
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
        search = load_pkl(os.path.join(s, f'search{seed_index}.pkl'))
        models.append(search)

    # Load dataset
    df = load_data(f'../data/formatted/{dataset_name}.csv')
    timestamps = load_data(f'../data/formatted/{dataset_name}_timestamps.csv')

    target = df.columns[-1]
    if "DEW_POINT_MEAN" in df.columns:
        df.rename(columns={'DEW_POINT_MEAN': 'dew_point'}, inplace=True)

    true_values = df[target].to_numpy()
    cocal_values = df['PM10_MEAN'].to_numpy()
    lon = df['LON_MEAN'].to_numpy()
    lat = df['LAT_MEAN'].to_numpy()
    coordinates = list(zip(lon, lat))

    if len(selected_features) == 0:
        df = df.copy()
    else:
        df = df[selected_features + [target]].copy()

    X = df.iloc[:, :-1]

    # Apply each model to the dataset and obtain prediction vectors
    predictions = []
    for actual_model in models:
        preds = actual_model.predict(X)
        predictions.append(preds)

    # Aggregate predictions by median
    median_preds = np.median(predictions, axis=0)

    # Compute Q1, Q3, and IQR
    q1 = np.percentile(predictions, 25, axis=0)
    q3 = np.percentile(predictions, 75, axis=0)

    return timestamps, coordinates, cocal_values, true_values, median_preds, q1, q3


# ======================================================================================
# SIMPLE ANALYSIS
# ======================================================================================

def mean_std_pm10_cocal_arpa(datasets):
    for dataset in datasets:
        df = load_data(f'../data/formatted/{dataset}.csv')
        cocal = df['PM10_MEAN'].values
        arpa = df['PM10_ARPA'].values
        print(dataset)
        print("COCAL ", np.mean(cocal), " ", np.std(cocal))
        print("ARPA ", np.mean(arpa), " ", np.std(arpa))

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


def print_latex_like_table_with_all_metrics_and_datasets(path, features, encoding, scaling, augmentation, models, test_size, n_iter, cv, linear_scaling, log_scale_target, n_train_records, seed_indexes):
    metrics = ["mae", "rmse", "r2"]
    datasets = ["volontari", "carpineto", "sincrotrone"]

    latex_string = ""

    all_model_median_scores = {}
    for model in models:
        if model in ('cocal_only', 'basic_median_delta'):
            s = create_dir_path_results(
                base_path=path,
                dataset='volontari',
                features=features,
                encoding='onehot',
                scaling='none',
                augmentation='none',
                model=model,
                test_size=test_size,
                n_iter=n_iter,
                cv=cv,
                linear_scaling=linear_scaling,
                log_scale_target=0,
                n_train_records=0,
            )
        else:
            s = create_dir_path_results(
                base_path=path,
                dataset='volontari',
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
        scores = {dataset: {metric: {'train': [], 'test': []} for metric in metrics} for dataset in datasets}

        for seed_index in seed_indexes:
            with open(os.path.join(s, f'result{seed_index}.json'), 'r') as f:
                res = json.load(f)
            for dataset in datasets:
                for metric in metrics:
                    scores[dataset][metric]['train'].append(res[dataset][metric]['train_score'])
                    scores[dataset][metric]['test'].append(res[dataset][metric]['test_score'])
        
        scores_median_values = {dataset: {metric: {'train': 0.0, 'test': 0.0} for metric in metrics} for dataset in datasets}
        for dataset in datasets:
            for metric in metrics:
                if scores[dataset][metric]['train']:
                    scores_median_values[dataset][metric]['train'] = round(statistics.median(scores[dataset][metric]['train']), 2)
                if scores[dataset][metric]['test']:
                    scores_median_values[dataset][metric]['test'] = round(statistics.median(scores[dataset][metric]['test']), 2)
        all_model_median_scores[model] = scores_median_values

    for model in models:
        scores_median_values = all_model_median_scores[model]
        model_string = ' '.join([only_first_char_upper(sss) for sss in (model.replace('_', ' ').capitalize() if len(model) > 3 else model.upper()).replace("Cocal", "COCAL").split(' ')])
        latex_string += f"{model_string} & "
        for dataset in datasets:
            for metric in metrics:
                if metric in ('mae', 'rmse'):
                    min_value_among_model = min([all_model_median_scores[m][dataset][metric]['test'] for m in models])
                    if scores_median_values[dataset][metric]['test'] == min_value_among_model:
                        latex_string += r'\bfseries '+ f"{scores_median_values[dataset][metric]['test']} & "
                    else:
                        latex_string += f"{scores_median_values[dataset][metric]['test']} & "
                else:  # r2
                    max_value_among_model = max([all_model_median_scores[m][dataset][metric]['test'] for m in models])
                    if scores_median_values[dataset][metric]['test'] == max_value_among_model:
                        latex_string += r'\bfseries '+ f"{scores_median_values[dataset][metric]['test']} & "
                    else:
                        latex_string += f"{scores_median_values[dataset][metric]['test']} & "
        latex_string = latex_string[:-2] + r" \\ " + "\n"
    
    print(latex_string)


# ======================================================================================
# PLOTS
# ======================================================================================

def create_and_draw_map_plot(path, model, features, selected_features, encoding, scaling, augmentation, test_size, n_iter, cv, linear_scaling, log_scale_target, n_train_records, seed_indexes, PLOT_ARGS):
    data = {dataset: None for dataset in ['volontari', 'carpineto', 'sincrotrone']}
    datasets = ['volontari', 'carpineto', 'sincrotrone']
    for dataset in datasets:
        timestamps, coordinates, cocal_values, true_values, aggregated_values, q1, q3 = apply_aggregated_model_from_repetitions(
            path=path,
            model=model,
            dataset_name=dataset,
            features=features,
            selected_features=selected_features,
            encoding=encoding,
            scaling=scaling,
            augmentation=augmentation,
            test_size=test_size,
            n_iter=n_iter,
            cv=cv,
            linear_scaling=linear_scaling,
            log_scale_target=log_scale_target,
            n_train_records=n_train_records,
            seed_indexes=seed_indexes
        )
        data[dataset] = (timestamps, coordinates, cocal_values, true_values, aggregated_values, q1, q3)
    model_string = ' '.join([only_first_char_upper(sss) for sss in (model.replace('_', ' ').capitalize() if len(model) > 3 else model.upper()).replace("Cocal", "COCAL").split(' ')])

    # fastplot
    plot = fastplot.plot(None, None, mode='callback',
                         callback=lambda plt: my_callback_function_that_actually_draws_map_plot(plt, data, model_string),
                         style='latex', **PLOT_ARGS)
    plot.savefig(f'map_plot.pdf', dpi=1200)
    plot.savefig(f'map_plot.png', dpi=1200)


def my_callback_function_that_actually_draws_map_plot(plt, data, model_name):
    # Compute vmin and vmax from all datasets and all values, including cocal, true, and aggregated
    vmin = float('inf')
    vmax = float('-inf')
    for _, (timestamps, coordinates, cocal_values, true_values, aggregated_values, q1, q3) in data.items():
        if (cocal_values is None) or (true_values is None) or (aggregated_values is None):
            continue
        if len(cocal_values) == 0 and len(true_values) == 0 and len(aggregated_values) == 0:
            continue
        all_values = np.concatenate([cocal_values, true_values, aggregated_values])
        if all_values.size:
            vmin = min(vmin, float(np.min(all_values)))
            vmax = max(vmax, float(np.max(all_values)))

    if vmin == float('inf') or vmax == float('-inf'):
        print("No data available to plot. Exiting map drawing.")
        return

    target = "PM10 Concentration (µg/m³)"

    # Figure and 3x4 grid (3 map columns + 1 colorbar column)
    fig = plt.figure(figsize=(15, 15), layout="constrained")
    gs = fig.add_gridspec(
        nrows=3,
        ncols=4,
        width_ratios=[1, 1, 1, 0.05],
        height_ratios=[1, 1, 1],
        wspace=0.02,
        hspace=0.02,
    )

    # Colorbar axis spans all rows in the 4th column
    cax = fig.add_subplot(gs[:, 3])

    # Column headers on the first row only
    col_labels = ["COCAL values", "Predicted values", "True values"]

    # Iterate rows (datasets)
    for i, (dataset, (timestamps, coordinates, cocal_values, true_values, aggregated_values, q1, q3)) in enumerate(data.items()):
        if coordinates is None or len(coordinates) == 0:
            print(f"No points to plot for dataset {dataset}. Skipping row.")
            continue
        if any(len(arr) == 0 for arr in [cocal_values, true_values, aggregated_values]):
            print(f"Missing values for dataset {dataset}. Skipping row.")
            continue
        if not (len(cocal_values) == len(true_values) == len(aggregated_values) == len(coordinates)):
            print(f"Mismatch between number of values and number of coordinates for dataset {dataset}. Skipping row.")
            continue

        # Axes for this row
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        ax3 = fig.add_subplot(gs[i, 2])

        # Build GeoDataFrame for projection
        gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lon, lat in coordinates], crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # Compute row extent (with padding) so all three subplots in the row share the same view
        xmin, ymin, xmax, ymax = gdf.total_bounds
        dx, dy = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05  # 5% padding
        extent = (xmin - dx, xmax + dx, ymin - dy, ymax + dy)

        # Helper to set up a single map axis
        def setup_ax(ax, values, label, dataset_name):
            gdf.plot(
                ax=ax,
                column=values,
                cmap="cividis",
                markersize=30,
                vmin=vmin,
                vmax=vmax,
                legend=False,
                alpha=0.7,
                edgecolor="k",
            )

            # Optional: draw outline polygon (if available)
            try:
                if not gdf.empty and len(gdf) > 2:
                    # union_all is available in newer geopandas; fallback to unary_union otherwise
                    hull_candidate = getattr(gdf, "union_all", None)
                    if callable(hull_candidate):
                        hull = gdf.union_all().convex_hull
                    else:
                        hull = gdf.unary_union.convex_hull
                    gpd.GeoSeries([hull], crs=gdf.crs).boundary.plot(
                        ax=ax, color="black", linewidth=0.6
                    )
            except Exception:
                # If hull computation fails, skip silently
                pass

            # Add basemap and lock extent
            try:
                ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)  # type: ignore
            except Exception:
                # Basemap is optional; continue if tiles fail to load
                pass
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            # Cosmetics
            if dataset_name == "sincrotrone":
                ax.set_xlabel("Longitude")
            if label == "COCAL values":
                ax.set_ylabel("Latitude")
            else:
                ax.tick_params(labelleft=False)

            # Column titles for the top row only
            if i == 0:
                ax.set_title(label)

            # Add row label to the last column only (right side)
            if label == "True values":
                axtwin = ax.twinx()
                axtwin.set_ylabel(dataset_name.capitalize(), rotation=270, labelpad=14)
                axtwin.yaxis.set_label_position("right")
                axtwin.tick_params(labelleft=False)
                axtwin.set_yticks([])
                axtwin.yaxis.tick_right()

            ax.grid(True, axis="both", which="major", color="gray", linestyle="--", linewidth=0.5)
            ax.tick_params(axis="both", which="both", reset=False, bottom=False, top=False, left=False, right=False)

        # Draw the three map types for this dataset row
        setup_ax(ax1, cocal_values, col_labels[0], dataset)
        setup_ax(ax2, aggregated_values, col_labels[1], dataset)
        setup_ax(ax3, true_values, col_labels[2], dataset)

    # Shared colorbar across all plots
    sm = plt.cm.ScalarMappable(cmap="cividis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label(target)

    # Suptitle for the model name
    fig.suptitle(f"{model_name}")

    return fig


def create_scatterplot_on_multiple_datasets_with_true_values_vs_predicted_values_from_aggregated_model(path, model, features, selected_features, encoding, scaling, augmentation, test_size, n_iter, cv, linear_scaling, log_scale_target, n_train_records, seed_indexes, PLOT_ARGS):
    data = {dataset: None for dataset in ['volontari', 'carpineto', 'sincrotrone']}
    datasets = ['volontari', 'carpineto', 'sincrotrone']
    for dataset in datasets:
        timestamps, coordinates, cocal_values, true_values, aggregated_values, q1, q3 = apply_aggregated_model_from_repetitions(
            path=path,
            model=model,
            dataset_name=dataset,
            features=features,
            selected_features=selected_features,
            encoding=encoding,
            scaling=scaling,
            augmentation=augmentation,
            test_size=test_size,
            n_iter=n_iter,
            cv=cv,
            linear_scaling=linear_scaling,
            log_scale_target=log_scale_target,
            n_train_records=n_train_records,
            seed_indexes=seed_indexes
        )
        data[dataset] = (true_values, aggregated_values, q1, q3)
    model_string = ' '.join([only_first_char_upper(sss) for sss in (model.replace('_', ' ').capitalize() if len(model) > 3 else model.upper()).replace("Cocal", "COCAL").split(' ')])

    # fastplot
    plot = fastplot.plot(None, None, mode='callback',
                         callback=lambda plt: my_callback_function_that_actually_draws_scatterplot_on_multiple_datasets_with_true_values_vs_predicted_values(plt, data, model_string),
                         style='latex', **PLOT_ARGS)
    plot.savefig(f'scatterplot.pdf', dpi=1200)
    plot.savefig(f'scatterplot.png', dpi=1200)

def my_callback_function_that_actually_draws_scatterplot_on_multiple_datasets_with_true_values_vs_predicted_values(plt, data, model_name):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), layout="constrained", squeeze=False)
    for i, (dataset, (true_values, aggregated_values, q1, q3)) in enumerate(data.items()):
        iqr = q3 - q1
        plt.sca(ax[0, i])
        plt.scatter(true_values, aggregated_values, color='red', label='Predictions')
        plt.errorbar(true_values, aggregated_values, yerr=iqr, fmt='o', color='red', ecolor='blue', elinewidth=1, capsize=2, alpha=0.5)
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='green', linestyle='--', label='Ideal', alpha=0.5)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{dataset.capitalize()}')
        plt.legend()
        plt.axis('equal')
        plt.xlim(min(true_values), max(true_values))
        plt.ylim(min(true_values), max(true_values))
        plt.gca().set_aspect('equal', adjustable='box')
        # Grid and ticks
        plt.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
        plt.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
        if i != 0:
            plt.ylabel('')
        # Explictly avoid to show 0 on axes
        if min(true_values) <= 0 <= max(true_values):
            plt.xticks([tick for tick in plt.xticks()[0] if tick != 0])
            plt.yticks([tick for tick in plt.yticks()[0] if tick != 0])
    # Overall title
    plt.suptitle(f'{model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def create_lineplot_on_dataset_with_arpa_vs_cocal_vs_aggregated_model_across_time(path, model, features, selected_features, encoding, scaling, augmentation, test_size, n_iter, cv, linear_scaling, log_scale_target, n_train_records, seed_indexes, PLOT_ARGS):
    data = {dataset: None for dataset in ['volontari', 'carpineto', 'sincrotrone']}
    datasets = ['volontari', 'carpineto', 'sincrotrone']

    for dataset in datasets:
        timestamps, coordinates, cocal_values, true_values, aggregated_values, q1, q3 = apply_aggregated_model_from_repetitions(
            path=path,
            model=model,
            dataset_name=dataset,
            features=features,
            selected_features=selected_features,
            encoding=encoding,
            scaling=scaling,
            augmentation=augmentation,
            test_size=test_size,
            n_iter=n_iter,
            cv=cv,
            linear_scaling=linear_scaling,
            log_scale_target=log_scale_target,
            n_train_records=n_train_records,
            seed_indexes=seed_indexes
        )
        data[dataset] = (timestamps, cocal_values, true_values, aggregated_values, q1, q3)

    # fastplot
    plot = fastplot.plot(None, None, mode='callback',
                         callback=lambda plt: my_callback_function_that_actually_draws_lineplot_on_dataset_with_arpa_vs_cocal_vs_aggregated_model_across_time(plt, data),
                         style='latex', **PLOT_ARGS)
    plot.savefig(f'lineplot.pdf', dpi=1200)
    plot.savefig(f'lineplot.png', dpi=1200)


def my_callback_function_that_actually_draws_lineplot_on_dataset_with_arpa_vs_cocal_vs_aggregated_model_across_time(plt, data):
    figsize = (12, 12)
    fig, ax = plt.subplots(3, 1, figsize=figsize, layout="constrained", squeeze=False)

    for i, (dataset, (timestamps, cocal_values, true_values, aggregated_values, q1, q3)) in enumerate(data.items()):
        x = np.arange(len(timestamps))

        if dataset == 'sincrotrone':
            ax[i, 0].set_xlabel("Timestamps")

        ax[i, 0].plot(x, true_values, label="ARPA", color='#2E8B57', linestyle='-', linewidth=1.4, markersize=10)
        ax[i, 0].plot(x, cocal_values, label="COCAL", color='#D77A1D', linestyle='--', linewidth=1.4, markersize=10)
        ax[i, 0].plot(x, aggregated_values, label="Aggregated Model", color='#625DF2', linestyle='-.', linewidth=1.4, markersize=10)
        ax[i, 0].fill_between(x, q1, q3, alpha=0.3, color='#625DF2')

        axtwin = ax[i, 0].twinx()
        axtwin.set_ylabel(dataset.capitalize(), rotation=270, labelpad=14)
        axtwin.yaxis.set_label_position("right")
        # Hide left labels on the twin axis; show ticks on the right only
        axtwin.tick_params(labelleft=False)
        # Remove y-ticks on the twin axis instead of setting empty ticklabels (avoids matplotlib warnings)
        axtwin.set_yticks([])
        axtwin.yaxis.tick_right()

        # Grid and ticks
        ax[i, 0].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
        ax[i, 0].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)

        ax[i, 0].set_xticks(np.arange(0, len(timestamps), max(1, len(timestamps)//10)))
        ax[i, 0].set_xticklabels([str(timestamps.iloc[i]['timestamp']) for i in range(0, len(timestamps), max(1, len(timestamps)//10))], rotation=45)
        ax[i, 0].set_ylabel("PM10 Concentration (µg/m³)")
        if dataset == 'volontari':
            ax[i, 0].set_title(f"PM10 Concentration Over Time")
        ax[i, 0].legend()


def my_callback_function_that_actually_draws_boxplot(plt, data, marked_models, dataset_split_palette, x_label="Model", y_label="MAE", title=None):
    figsize = (10, 10)
    
    fig, ax = plt.subplots(3, 1, figsize=figsize, layout='constrained', squeeze=False)
    zones = ["volontari", "carpineto", "sincrotrone"]
    for i, zone in enumerate(zones):
        
        if zone == 'sincrotrone':
            ax[i, 0].set_xlabel(x_label)
        ax[i, 0].set_ylabel(y_label)

        axtwin = ax[i, 0].twinx()
        axtwin.set_ylabel(zone.capitalize(), rotation=270, labelpad=14)
        axtwin.yaxis.set_label_position("right")
        # Hide left labels on the twin axis; show ticks on the right only
        axtwin.tick_params(labelleft=False)
        # Remove y-ticks on the twin axis instead of setting empty ticklabels (avoids matplotlib warnings)
        axtwin.set_yticks([])
        axtwin.yaxis.tick_right()

        # Grid and ticks
        ax[i, 0].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
        ax[i, 0].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)

        if title and zone == 'volontari':
            ax[i, 0].set_title(title)

        # Hide x-tick labels for certain zones
        if zone == 'volontari' or zone == 'carpineto':
            ax[i, 0].tick_params(labelbottom=False)
            ax[i, 0].set_xticklabels([])

        # Draw boxplot
        sns.boxplot(data=data[zone], x="Model", y="MAE", hue="Dataset", palette=dataset_split_palette, legend=zone == 'volontari',
                    log_scale=None, fliersize=0.0, showfliers=False, ax=ax[i, 0])

        # Annotate stars for marked models
        ymin, _ = ax[i, 0].get_ylim()
        star_y = ymin + (0.027 * (ax[i, 0].get_ylim()[1] - ymin))  # a bit below ymin

        # Determine tick positions and labels robustly (works even if labels are hidden)
        tick_positions = ax[i, 0].get_xticks()
        tick_texts = [t.get_text() for t in ax[i, 0].get_xticklabels()]
        # If labels are empty (hidden), fallback to model names from the data preserving order
        if not any(lbl.strip() for lbl in tick_texts):
            try:
                tick_labels = list(pd.unique(data[zone]['Model']))
            except Exception:
                # Fallback to numeric positions if something goes wrong
                tick_labels = [str(int(i)) for i in range(len(tick_positions))]
        else:
            tick_labels = tick_texts

        # Use positions (not just indices) so text is placed correctly even when labels are hidden
        for pos, model_label in zip(tick_positions, tick_labels):
            for key in marked_models[zone]:
                # Check if this label corresponds to this key
                if model_label == key:
                    if marked_models[zone][key]:
                        ax[i, 0].text(
                            pos, star_y, r"\textbf{*}",
                            ha='center', va='top', fontsize=14, color='black', clip_on=False
                        )
                    break

        
        

def export_mae_boxplot(path, features, encoding, scaling, augmentation, models, test_size, n_iter, cv,
                        linear_scaling, log_scale_target, n_train_records, seed_indexes,
                        dataset_split_palette, title, dpi, PLOT_ARGS):
    datasets = ["volontari", "carpineto", "sincrotrone"]
    all_data = {zone: [] for zone in datasets}
    model_values = {zone: {} for zone in datasets}

    for zone in datasets:
        for model in models:
            if model in ('cocal_only', 'basic_median_delta'):
                model_path = create_dir_path_results(
                    base_path=path,
                    dataset='volontari',
                    features=features,
                    encoding='onehot',
                    scaling='none',
                    augmentation='none',
                    model=model,
                    test_size=test_size,
                    n_iter=n_iter,
                    cv=cv,
                    linear_scaling=linear_scaling,
                    log_scale_target=0,
                    n_train_records=0,
                )
            else:
                model_path = create_dir_path_results(
                    base_path=path,
                    dataset='volontari',
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
                if zone == "volontari":
                    all_data[zone].append({"Model": model_string, "Dataset": "Train", "MAE": train_mae})
                all_data[zone].append({"Model": model_string, "Dataset": "Test", "MAE": test_mae})

            model_values[zone][model] = temp

    marked_models = {}

    for zone in datasets:
        holm, mann = perform_mannwhitneyu_holm_bonferroni(model_values[zone], alternative='less')
        marked_models[zone] = {}
        for model in models:
            model_string = ' '.join([only_first_char_upper(sss) for sss in (model.replace('_', ' ').capitalize() if len(model) > 3 else model.upper()).replace("Cocal", "COCAL").split(' ')])

            marked_models[zone][model_string] = False
            if holm[model]:
                marked_models[zone][model_string] = True
            else:
                temp = mann[model]
                for key in temp:
                    if temp[key]:
                        marked_models[zone][model_string] = True
                        break

    all_df = {}
    for zone in datasets:
        all_df[zone] = pd.DataFrame(all_data[zone])

    plot = fastplot.plot(None, None, mode='callback',
                         callback=lambda plt: my_callback_function_that_actually_draws_boxplot(plt, all_df, marked_models, dataset_split_palette, title=title),
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
    models = ["cocal_only", "basic_median_delta", "linear", "elasticnet", "decision_tree", "symbolic_regression", "svr", "random_forest", "bagging", "gradient_boosting", "adaboost", "mlp"]
    #models = ["cocal_only", "basic_median_delta", "linear", "elasticnet", "decision_tree", "symbolic_regression"]
    #models = ["elasticnet", "svr", "random_forest", "bagging", "gradient_boosting", "adaboost", "mlp"]

    #print_basic_scores_with_cap(None, None, None, path=path, dataset=dataset, test_dataset=None, features=features, encoding=encoding, scaling=scaling, augmentation=augmentation, models=models, test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling, log_scale_target=log_scale_target, n_train_records=n_train_records, seed_indexes=seed_indexes)
    #mean_std_pm10_cocal_arpa(['volontari', 'carpineto', 'sincrotrone'])
    #export_mae_boxplot(path=path, title='Black-Box Methods', features=features, encoding=encoding, scaling=scaling, augmentation=augmentation, models=models, test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling, log_scale_target=log_scale_target, n_train_records=n_train_records, seed_indexes=seed_indexes, dataset_split_palette=dataset_split_palette, dpi=1200, PLOT_ARGS=PLOT_ARGS)
    #collect_mae_lineplot_data(path=path, dataset=dataset, features=features, encoding=encoding, scaling=scaling, augmentation=augmentation, model='gradient_boosting', test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling, log_scale_target=log_scale_target, n_train_record_list=[400, 800, 1200, 1600, 2000, 0], seed_indexes=seed_indexes, dataset_split_palette=dataset_split_palette, dpi=500, PLOT_ARGS=PLOT_ARGS)

    #print_formula_sr(path=path, dataset=dataset, features=features, selected_features=selected_features, encoding=encoding, scaling=scaling, augmentation=augmentation, test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling,
    #                 log_scale_target=log_scale_target, n_train_records=n_train_records, seed_index=12)

    #print_latex_like_table_with_all_metrics_and_datasets(path=path, features=features, encoding=encoding, scaling=scaling, augmentation=augmentation, models=models, test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling, log_scale_target=log_scale_target, n_train_records=n_train_records, seed_indexes=seed_indexes)

    #create_lineplot_on_dataset_with_arpa_vs_cocal_vs_aggregated_model_across_time(path=path, model='symbolic_regression', features=features, selected_features=selected_features, encoding=encoding, scaling=scaling, augmentation=augmentation, test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling, log_scale_target=log_scale_target, n_train_records=n_train_records, seed_indexes=seed_indexes, PLOT_ARGS=PLOT_ARGS)

    #create_scatterplot_on_multiple_datasets_with_true_values_vs_predicted_values_from_aggregated_model(path=path, model='symbolic_regression', features=features, selected_features=selected_features, encoding=encoding, scaling=scaling, augmentation=augmentation, test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling, log_scale_target=log_scale_target, n_train_records=n_train_records, seed_indexes=seed_indexes, PLOT_ARGS=PLOT_ARGS)

    create_and_draw_map_plot(path=path, model='symbolic_regression', features=features, selected_features=selected_features, encoding=encoding, scaling=scaling, augmentation=augmentation, test_size=test_size, n_iter=n_iter, cv=cv, linear_scaling=linear_scaling, log_scale_target=log_scale_target, n_train_records=n_train_records, seed_indexes=seed_indexes, PLOT_ARGS=PLOT_ARGS)


if __name__ == '__main__':
    main()
