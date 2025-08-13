from tsairpoll.data_loader import *
from tsairpoll.eda import *
from tsairpoll.model_evaluation import evaluate_model
from tsairpoll.model_training import preprocess_and_train


def quick():
    #merge_arpa_csv_files('data/raw/arpa/pm10_volontari_2022_2023_2024/', 'data/raw/arpa/pm10_volontari_2022_2023_2024/arpa.csv')
    #merge_cocal_csv_files('data/raw/cocal/pm10_volontari_2022_2023_2024/', 'data/raw/cocal/pm10_volontari_2022_2023_2024/cocal.csv')
    df_merged, timestamps = integrate_arpa_cocal(
        'data/raw/arpa/elenco_centraline_trieste.csv',
        'data/raw/arpa/2024/pm10/carpineto/arpa.csv',
        'data/raw/cocal/2024/carpineto/cocal.csv',
        station_name="Trieste - via Carpineto",
        parameter_name="Particelle sospese PM10",
        max_distance=100,
        device_type='dynamic',
        use_delta_as_target=False
    )
    formatted_file_path = "data/formatted/carpineto.csv"  # Update with actual path
    save_csv_data(df_merged, formatted_file_path)
    save_csv_data(timestamps, "data/formatted/carpineto_timestamps.csv")

    # df = load_data('data/formatted/volontari.csv')
    # train_df, train_timestamp, test_df, test_timestamp = split_data_from_path('data/formatted/volontari.csv', 'data/formatted/volontari_timestamps.csv', 0.2, 42)
    # save_csv_data(train_df, 'data/formatted/volontari_real.csv')
    # save_csv_data(test_df, 'data/formatted/volontari_eda.csv')
    # save_csv_data(train_timestamp, 'data/formatted/volontari_real_timestamps.csv')
    # save_csv_data(test_timestamp, 'data/formatted/volontari_eda_timestamps.csv')
    # perform_eda(load_data('data/formatted/volontari_eda.csv'))
    # FEATURES: "season","weekend","time_of_day","device_type","gps_kmh","gps_dir","gps_alt","BM_H","BM_P","BM_T","DS_T","PM10","PM10_ARPA"
    # selected_features = ["time_of_day", "gps_kmh", "BM_H", "BM_P", "BM_T", "DS_T", "PM10"]
    #
    # search, X_train, X_test, y_train, y_test, train_timestamp, test_timestamp = preprocess_and_train(
    #     load_data('data/formatted/volontari_real.csv'),
    #     load_data('data/formatted/volontari_real_timestamps.csv'),
    #     selected_features,
    #     'onehot',
    #     'robust',
    #     'bagging',
    #     0.3,
    #     10,
    #     5,
    #     42,
    #     1,
    #     False,
    #     False,
    #     0
    # )
    # score_dict = evaluate_model(search, X_train, X_test, y_train, y_test)
    # print(score_dict['train_score'])
    # print(score_dict['test_score'])
    #plot_three_lines(list(range(1, len(test_idx) + 1)), y_test, score_dict['test_pred'], X_test.values[:, -1].flatten(), ['arpa', 'estimated', 'cocal'])


if __name__ == '__main__':
    quick()
