import pandas as pd
import numpy as np
import time
import sys, getopt
from nilmtk import DataSet
import warnings
import multiprocessing

cores = multiprocessing.cpu_count()  # Number of CPU cores on your system
partitions = cores  # Define as many partitions as you want


if not sys.warnoptions:
    warnings.simplefilter("ignore")

work_directory = "/home/bruno.strasser/work/"

dataset_ukdale_path = work_directory + "ukdale.h5"
dataset_redd_path = work_directory + "redd.h5"
dataset_iawe_path = work_directory + "iawe.h5"
# dataset_greend_path = work_directory + "greend.h5"
dataset_combed_path = work_directory + "combed.h5"

# number windows to extract foreach dataset
k = 1000

appliances = ['dish washer', 'fridge', 'kettle', 'microwave', 'washing machine']

activations_csv = work_directory + "activations.csv"
windows_csv = work_directory + "windows_data.csv"
date_ranges_csv = work_directory + "date_ranges.csv"

activations_params = {'max_power': {}, 'on_power_threshold': {}, 'min_on_duration': {}, 'min_off_duration': {}}
activations_params['max_power']['dish washer'] = 2500
activations_params['on_power_threshold']['dish washer'] = 1800
activations_params['min_on_duration']['dish washer'] = 1800
activations_params['min_off_duration']['dish washer'] = 10
activations_params['max_power']['fridge'] = 300
activations_params['on_power_threshold']['fridge'] = 12
activations_params['min_on_duration']['fridge'] = 60
activations_params['min_off_duration']['fridge'] = 50
activations_params['max_power']['kettle'] = 3100
activations_params['on_power_threshold']['kettle'] = 0
activations_params['min_on_duration']['kettle'] = 12
activations_params['min_off_duration']['kettle'] = 2000
activations_params['max_power']['microwave'] = 3000
activations_params['on_power_threshold']['microwave'] = 30
activations_params['min_on_duration']['microwave'] = 12
activations_params['min_off_duration']['microwave'] = 200
activations_params['max_power']['washing machine'] = 2500
activations_params['on_power_threshold']['washing machine'] = 160
activations_params['min_on_duration']['washing machine'] = 1800
activations_params['min_off_duration']['washing machine'] = 20

windows_size = {'dish washer': 120 * 60,
                'fridge': 15 * 60,
                'kettle': 15 * 60,
                'microwave': 15 * 60,
                'washing machine': 30 * 60
                }

windows_max_offset = {
    'dish washer': 6,
    'fridge': 3,
    'kettle': 3,
    'microwave': 3,
    'washing machine': 3
}

windows_period = {
    'dish washer': 120,
    'fridge': 30,
    'kettle': 30,
    'microwave': 30,
    'washing machine': 60
}

windows_df_files = {
    'dish washer': work_directory + 'win_dish_washer.csv',
    'fridge': work_directory + 'win_fridge.csv',
    'kettle': work_directory + 'win_kettle.csv',
    'microwave': work_directory + 'win_microwave.csv',
    'washing machine': work_directory + 'win_washing_machine.csv'
}

windows_df_data_files = {
    'dish washer': work_directory + 'windat_dish_washer.csv',
    'fridge': work_directory + 'windat_fridge.csv',
    'kettle': work_directory + 'windat_kettle.csv',
    'microwave': work_directory + 'windat_microwave.csv',
    'washing machine': work_directory + 'windat_washing_machine.csv'
}


def load_datasets():
    datasets = {
        'redd': DataSet(dataset_redd_path),
        'ukdale': DataSet(dataset_ukdale_path)
        # 'iawe': DataSet(dataset_iawe_path)
        # 'greend': DataSet(dataset_greend_path)
        # 'combed': DataSet(dataset_combed_path)
    }
    return datasets


def generate_time_series(datasets, app, freq):
    for dataset in datasets:
        for b in [x + 1 for x in range(len(datasets[dataset].buildings))]:
            try:
                elec = datasets[dataset].buildings[b].elec
                df_main = elec.mains().power_series(ac_type='apparent').__next__()
                df_main = df_main.resample('{}S'.format(freq)).mean()
                df_main.to_csv(work_directory + "{}_{}_{}_main.csv".format(dataset, b, app))
                del df_main
                df_main = elec.mains().power_series(ac_type='apparent').__next__()
                df_main.to_csv(work_directory + "{}_{}_{}_main_raw.csv".format(dataset, b, app))
                del df_main
                df_app = elec[app].power_series(ac_type='active').__next__()
                df_app = df_app.resample('{}S'.format(freq)).mean()
                df_app.to_csv(work_directory + "{}_{}_{}.csv".format(dataset, b, app))
                del df_app
                df_app = elec[app].power_series(ac_type='active').__next__()
                df_app.to_csv(work_directory + "{}_{}_{}_raw.csv".format(dataset, b, app))
                del df_app
            except KeyError:
                continue


if __name__ == '__main__':

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "a:f", ["prefix=", "app=", "freq="])
    app = None
    freq = None
    work_directory = ""
    output_data = "win_data.csv"
    output_win = "win.csv"
    for opt, arg in opts:
        if opt == "--prefix":
            work_directory = arg
        if opt in ("-a", "--app"):
            app = arg
        if opt in ("-f", "--freq"):
            freq = arg

    if app is None:
        exit()
    if freq is None:
        exit()

    datasets = load_datasets()
    generate_time_series(datasets, app, freq)
    exit()

