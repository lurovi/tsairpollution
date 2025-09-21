import cProfile
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import threading
import zlib
import time
import traceback
import json
import argparse
import warnings

from tsairpoll.model_evaluation import evaluate_model
from tsairpoll.model_training import preprocess_and_train

from sklearn.exceptions import ConvergenceWarning
from tsairpoll.data_loader import load_data, create_dir_path_results, dataframe_filtered_to_numpy

from tsairpoll.utils import is_valid_filename, save_pkl

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


completed_csv_lock = threading.Lock()


def main():
    run_with_exceptions_path: str = 'run_with_exceptions/'

    if not os.path.isdir(run_with_exceptions_path):
        os.makedirs(run_with_exceptions_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train a model on air pollution data.")

    # Command-line arguments
    parser.add_argument("--data", type=str, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--features", type=str, required=True, help="List of selected feature names separated by comma.")
    parser.add_argument("--encoding", type=str, required=True, help="Encoding type for categorical features.")
    parser.add_argument("--scaling", type=str, required=True, help="Scaling strategy for numerical features.")
    parser.add_argument("--augmentation", type=str, required=True, help="Data augmentation strategy for training with imbalanced target.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'random_forest', 'linear_regression').")
    parser.add_argument("--test_size", type=float, required=True, help="Proportion of the dataset to use for testing.")
    parser.add_argument("--n_iter", type=int, required=True, help="Number of iterations for hyperparameter search.")
    parser.add_argument("--cv", type=int, required=True, help="Number of cross-validation folds.")
    parser.add_argument("--linear_scaling", type=int, required=True, help="Whether or not performing linear scaling.")
    parser.add_argument("--log_scale_target", type=int, required=True, help="Whether or not log-scaling target for regularization.")
    parser.add_argument("--n_train_records", type=int, required=True, help="Number of training records.")
    parser.add_argument("--seed_index", type=int, required=True, help="Random seed_index for reproducibility.")
    parser.add_argument("--run_id", type=str, default='default', help="The run id, used for logging purposes of successful runs.")
    parser.add_argument("--verbose", required=False, action="store_true", help="Verbose flag.")
    parser.add_argument("--profile", required=False, action="store_true", help="Whether to run and log profiling of code or not.")

    args = parser.parse_args()

    data = args.data
    features = args.features
    encoding = args.encoding
    scaling = args.scaling
    augmentation = args.augmentation
    model = args.model
    test_size = float(args.test_size)
    n_iter = int(args.n_iter)
    cv = int(args.cv)
    linear_scaling = int(args.linear_scaling)
    log_scale_target = int(args.log_scale_target)
    n_train_records = int(args.n_train_records)
    seed_index = int(args.seed_index)
    run_id: str = args.run_id

    verbose: int = int(args.verbose)
    profiling: int = int(args.profile)

    args_string = ";".join(f"{key};{vars(args)[key]}" for key in sorted(list(vars(args).keys())) if key not in ('profile', 'verbose'))
    all_items_string = ";".join(f"{key}={value}" for key, value in vars(args).items())

    pr = None
    if profiling != 0:
        pr = cProfile.Profile()
        pr.enable()

    try:
        if not is_valid_filename(run_id):
            raise ValueError(f'run_id {run_id} is not a valid filename.')

        if seed_index < 1:
            raise AttributeError(f'seed_index does not start from 1, it is {seed_index}.')

        timestamps = data.replace('.csv', '_timestamps.csv')
        name_without_ext = os.path.splitext(os.path.basename(data))[0]
        features_for_the_results_file = f'features{zlib.adler32(bytes(features, "utf-8"))}'

        with open('random_seeds.txt', 'r') as f:
            # THE ACTUAL SEED TO BE USED IS LOCATED AT POSITION seed_index - 1 SINCE seed_index IS AN INDEX THAT STARTS FROM 1
            all_actual_seeds = [int(curr_actual_seed_as_str) for curr_actual_seed_as_str in f.readlines()]
        seed = all_actual_seeds[seed_index - 1]

        # Load datasets
        df = load_data(data)
        df_timestamps = load_data(timestamps)

        carpineto = load_data("data/formatted/carpineto.csv")
        carpineto_timestamps = load_data("data/formatted/carpineto_timestamps.csv")
        sincrotrone = load_data("data/formatted/sincrotrone.csv")
        sincrotrone_timestamps = load_data("data/formatted/sincrotrone_timestamps.csv")

        selected_features = features.split(',')

        X_carpineto, y_carpineto = dataframe_filtered_to_numpy(carpineto, carpineto_timestamps, selected_features)
        X_sincrotrone, y_sincrotrone = dataframe_filtered_to_numpy(sincrotrone, sincrotrone_timestamps, selected_features)

        # Call training function
        start_time = time.time()
        search, X_train, X_test, y_train, y_test, train_timestamp, test_timestamp = preprocess_and_train(
            df=df,
            timestamps=df_timestamps,
            selected_features=selected_features,
            encoding_type=encoding,
            scaling_strategy=scaling,
            augmentation_mode=augmentation,
            model_name=model,
            test_size=test_size,
            n_iter=n_iter,
            cv=cv,
            seed=seed,
            linear_scaling=linear_scaling != 0,
            log_scale_target=log_scale_target != 0,
            n_train_records=n_train_records,
            verbose=verbose
        )
        end_time = time.time()
        execution_time_in_minutes = (end_time - start_time) * (1 / 60)

        metrics = ['r2', 'mae', 'mse', 'rmse']
        score_dict = {k: evaluate_model(search, X_train, X_test, y_train, y_test, k) for k in metrics}
        score_dict_carpineto = {k: evaluate_model(search, X_train, X_carpineto, y_train, y_carpineto, k) for k in metrics}
        score_dict_sincrotrone = {k: evaluate_model(search, X_train, X_sincrotrone, y_train, y_sincrotrone, k) for k in metrics}

        final_path = create_dir_path_results(
            'results/',
            dataset=name_without_ext,
            features=features_for_the_results_file,
            encoding=encoding,
            scaling=scaling,
            model=model,
            augmentation=augmentation,
            test_size=test_size,
            n_iter=n_iter,
            cv=cv,
            linear_scaling=linear_scaling,
            log_scale_target=log_scale_target,
            n_train_records=n_train_records,
        )
        if not os.path.isdir(final_path):
            os.makedirs(final_path, exist_ok=True)

        result_dict = {'seed': seed, 'seed_index': seed_index,
                       'selected_features': selected_features,
                       'data': data, 'timestamps': timestamps, 'run_id': run_id,
                       'execution_time_in_minutes': execution_time_in_minutes,
                       name_without_ext: {}, 'carpineto': {}, 'sincrotrone': {}}

        for k in metrics:
            # Main dataset
            train_score, test_score = score_dict[k]['train_score'], score_dict[k]['test_score']
            train_score_sum_cocal, test_score_sum_cocal = score_dict[k]["train_score_sum_cocal"], score_dict[k]["test_score_sum_cocal"]

            result_dict[name_without_ext][k] = {'train_score': train_score, 'test_score': test_score,
                                                'train_score_sum_cocal': train_score_sum_cocal, 'test_score_sum_cocal': test_score_sum_cocal}
            # Carpineto
            train_score, test_score = score_dict_carpineto[k]['train_score'], score_dict_carpineto[k]['test_score']
            train_score_sum_cocal, test_score_sum_cocal = score_dict_carpineto[k]["train_score_sum_cocal"], score_dict_carpineto[k]["test_score_sum_cocal"]

            result_dict['carpineto'][k] = {'train_score': train_score, 'test_score': test_score,
                                                'train_score_sum_cocal': train_score_sum_cocal,
                                                'test_score_sum_cocal': test_score_sum_cocal}
            # Sincrotrone
            train_score, test_score = score_dict_sincrotrone[k]['train_score'], score_dict_sincrotrone[k]['test_score']
            train_score_sum_cocal, test_score_sum_cocal = score_dict_sincrotrone[k]["train_score_sum_cocal"], score_dict_sincrotrone[k]["test_score_sum_cocal"]

            result_dict['sincrotrone'][k] = {'train_score': train_score, 'test_score': test_score,
                                                'train_score_sum_cocal': train_score_sum_cocal,
                                                'test_score_sum_cocal': test_score_sum_cocal}

        if model == 'symbolic_regression':
            result_dict['latex'] = str(search.best_estimator_.pipeline['regressor'].latex())
            result_dict['sympy'] = str(search.best_estimator_.pipeline['regressor'].sympy())
            result_dict['latex_table'] = str(search.best_estimator_.pipeline['regressor'].latex_table())

        if model == 'GP-GOMEA':
            result_dict['sympy'] = str(search.best_estimator_.pipeline['regressor'].get_model().replace("p/", "/").replace("plog", "log"))
            #result_dict['sympy'] = str(search.best_estimator_.pipeline['regressor'].model)

        with open(os.path.join(final_path, f'result{seed_index}.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)

        save_pkl(search, os.path.join(final_path, f'search{seed_index}.pkl'))

        with completed_csv_lock:
            with open(os.path.join('results/', f'completed_{run_id}.txt'), 'a+') as terminal_std_out:
                terminal_std_out.write(args_string)
                terminal_std_out.write('\n')
            print(f'Completed run: {all_items_string}.')
    except Exception as e:
        try:
            error_string = str(traceback.format_exc())
            with open(os.path.join(run_with_exceptions_path, f'error_{zlib.adler32(bytes(args_string, "utf-8"))}.txt'), 'w') as f:
                f.write(args_string + '\n\n' + all_items_string + '\n\n' + error_string)
            print(f'\nException in run: {all_items_string}.\n\n{str(e)}\n\n')
        except Exception as ee:
            with open(os.path.join(run_with_exceptions_path, 'error_in_error.txt'), 'w') as f:
                f.write(str(traceback.format_exc()) + '\n\n')
            print(str(ee))

    if profiling != 0:
        pr.disable()
        pr.print_stats(sort='tottime')


if __name__ == '__main__':
    main()
