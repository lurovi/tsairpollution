from tsairpoll.data_loader import *
from tsairpoll.eda import *
from tsairpoll.model_evaluation import evaluate_model
from tsairpoll.model_training import preprocess_and_train


def quick():
    # for zone in ['volontari', 'carpineto', 'sincrotrone']:
    #     merge_arpa_csv_files(f'data_pre/raw/arpa/2025/pm10/{zone}/', f'data_pre/raw/arpa/2025/pm10/{zone}/arpa.csv')
    #     merge_cocal_csv_files(f'data_pre/raw/cocal/2025/{zone}/', f'data_pre/raw/cocal/2025/{zone}/cocal.csv')
    # quit()
    # for zone in ['volontari', 'carpineto', 'sincrotrone']:
    #     dict_zone_station = {'volontari': "Trieste - P.zza Volontari Giuliani",
    #                          'carpineto': "Trieste - via Carpineto",
    #                          'sincrotrone': "Trieste - Sincrotrone"}
    #     df_merged, timestamps = integrate_arpa_cocal(
    #         'data_pre/raw/arpa/elenco_centraline_trieste.csv',
    #         f'data_pre/raw/arpa/2025/pm10/{zone}/arpa.csv',
    #         f'data_pre/raw/cocal/2025/{zone}/cocal.csv',
    #         station_name=dict_zone_station[zone],
    #         parameter_name="Particelle sospese PM10",
    #         max_distance=100,
    #         device_type='dynamic',
    #         use_delta_as_target=False
    #     )
    #     formatted_file_path = f"data_pre/formatted/2025/{zone}.csv"  # Update with actual path
    #     save_csv_data(df_merged, formatted_file_path)
    #     save_csv_data(timestamps, f"data_pre/formatted/2025/{zone}_timestamps.csv")

    # df = load_data('data_pre/formatted/2025/volontari.csv')
    # train_df, train_timestamp, test_df, test_timestamp = split_data_from_path('data/formatted/volontari.csv', 'data/formatted/volontari_timestamps.csv', 0.2, 42)
    # save_csv_data(train_df, 'data/formatted/volontari_real.csv')
    # save_csv_data(test_df, 'data/formatted/volontari_eda.csv')
    # save_csv_data(train_timestamp, 'data/formatted/volontari_real_timestamps.csv')
    # save_csv_data(test_timestamp, 'data/formatted/volontari_eda_timestamps.csv')
    #perform_eda(load_data('data_pre/formatted/2025/volontari.csv'),
    #            ["month", "day_of_week", "week_of_month", "season", "weekend", "time_of_day", "dew_point", "NUM_MEASUREMENTS", "DS_T_MEAN", "BM_T_MEAN", "BM_H_MEAN", "BM_P_MEAN", "GPS_KMH_MEAN", "GPS_DIR_MEAN", "GPS_ALT_MEAN", "PM10_MEAN", "PM10_ARPA"])
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
    # plot_three_lines(list(range(1, len(test_idx) + 1)), y_test, score_dict['test_pred'], X_test.values[:, -1].flatten(), ['arpa', 'estimated', 'cocal'])

    # concatenate rows from a list of csv files into a single csv file
    l = [load_data('data/formatted/2025/volontari.csv'), load_data('data/formatted/2025/carpineto.csv'), load_data('data/formatted/2025/sincrotrone.csv')]
    merged_df = pd.concat(l, ignore_index=True)
    save_csv_data(merged_df, 'data/formatted/2025/merged_zones.csv')

if __name__ == '__main__':
    quick()
