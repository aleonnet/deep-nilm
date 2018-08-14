import datetime as dt
import traceback
import pandas as pd
import numpy as np
import random
import time
import sys, getopt
import os
import warnings
import multiprocessing
import copy

sys.setrecursionlimit(10000)

np.warnings.filterwarnings('ignore')

cores = multiprocessing.cpu_count()  # Number of CPU cores on your system
partitions = cores  # Define as many partitions as you want


def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
                                  for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))


if not sys.warnoptions:
    warnings.simplefilter("ignore")

work_directory = "/home/bruno.strasser/work"

dataset_ukdale_path = work_directory + "/ukdale.h5"
dataset_redd_path = work_directory + "/redd.h5"
dataset_iawe_path = work_directory + "/iawe.h5"
# dataset_greend_path = work_directory + "/greend.h5"
dataset_combed_path = work_directory + "/combed.h5"

# number windows to extract foreach dataset
k = 1000

appliances = ['dish washer', 'fridge', 'kettle', 'microwave', 'washing machine']

activations_csv = work_directory + "/activations.csv"
windows_csv = work_directory + "/windows_data.csv"
date_ranges_csv = work_directory + "/date_ranges.csv"

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

#windows_size = {'dish washer': 120 * 60,
#                'fridge': 15 * 60,
#                'kettle': 15 * 60,
#                'microwave': 15 * 60,
#                'washing machine': 30 * 60
#                }

windows_max_offset = {
    'dish washer': 6 * 60,
    'fridge': 3 * 60,
    'kettle': 3 * 60,
    'microwave': 3 * 60,
    'washing machine': 3 * 60
}

windows_period = {
    'dish washer': 120,
    'fridge': 30,
    'kettle': 30,
    'microwave': 30,
    'washing machine': 60
}

windows_df_files = {
    'dish washer': work_directory + '/win_dish_washer.csv',
    'fridge': work_directory + '/win_fridge.csv',
    'kettle': work_directory + '/win_kettle.csv',
    'microwave': work_directory + '/win_microwave.csv',
    'washing machine': work_directory + '/win_washing_machine.csv'
}

windows_df_data_files = {
    'dish washer': work_directory + '/windat_dish_washer.csv',
    'fridge': work_directory + '/windat_fridge.csv',
    'kettle': work_directory + '/windat_kettle.csv',
    'microwave': work_directory + '/windat_microwave.csv',
    'washing machine': work_directory + '/windat_washing_machine.csv'
}


def parse_dates_date_ranges(date_ranges, datasets):
    for dataset in datasets:
        for b in [x + 1 for x in range(len(datasets[dataset].buildings))]:
            for app in appliances:
                try:
                    date_ranges[dataset][str(b)][app]['start'] = pd.to_datetime(
                        date_ranges[dataset][str(b)][app]['start_str'])
                    date_ranges[dataset][str(b)][app]['end'] = pd.to_datetime(
                        date_ranges[dataset][str(b)][app]['end_str'])
                except KeyError:
                    continue
    return date_ranges


def save_date_ranges(date_ranges, datasets, appliances):
    date_ranges_columns = ['dataset', 'building', 'appliance', 'start', 'end']
    df = pd.DataFrame(columns=date_ranges_columns)
    for dataset in datasets:
        for b in [x + 1 for x in range(len(datasets[dataset].buildings))]:
            for app in appliances:
                try:
                    df = df.append(pd.DataFrame([[dataset, str(b), app,
                                                  str(date_ranges[dataset][str(b)][app]['start']),
                                                  str(date_ranges[dataset][str(b)][app]['end'])]],
                                                columns=date_ranges_columns),
                                   ignore_index=True)
                except KeyError:
                    continue
    df.to_csv(date_ranges_csv, quotechar="'", columns=date_ranges_columns, encoding='utf-8', index=False)


def load_date_ranges(app):
    df = pd.read_csv(date_ranges_csv, sep=',', header=0, encoding='utf-8',
                     names=['dataset', 'building', 'appliance', 'start', 'end'],
                     date_parser=pd.to_datetime,
                     parse_dates=['start', 'end'])
    if app is not None:
        df = df[df['appliance'] == app]
    return df


def generate_time_series(datasets, appliances):
    for dataset in datasets:
        for b in [x + 1 for x in range(len(datasets[dataset].buildings))]:

            for app in appliances:
                try:
                    elec = datasets[dataset].buildings[b].elec
                    df_main = elec.mains().power_series(ac_type='active').__next__()
                    df_main.to_csv(work_directory + "/{}_{}_{}_main.csv".format(dataset, b, app))
                    df_app = elec[app].power_series(ac_type='apparent').__next__()
                    df_app.to_csv(work_directory + "/{}_{}_{}.csv".format(dataset, b, app))

                except KeyError:
                    continue

def generate_date_ranges(datasets, appliances):
    date_ranges = {}
    for dataset in datasets:
        date_ranges[dataset] = {}
        for b in [x + 1 for x in range(len(datasets[dataset].buildings))]:
            date_ranges[dataset][str(b)] = {}
            df_main = datasets[dataset].buildings[b].elec.mains().power_series().__next__()
            start_main = df_main.first_valid_index()
            end_main = df_main.last_valid_index()
            df_main = None
            for app in appliances:
                try:
                    df_app = datasets[dataset].buildings[b].elec[app].power_series().__next__()
                    start_app = df_app.first_valid_index()
                    end_app = df_app.last_valid_index()
                except KeyError:
                    continue
                date_ranges[dataset][str(b)][app] = {}
                date_ranges[dataset][str(b)][app]['start'] = max(start_app, start_main)
                date_ranges[dataset][str(b)][app]['end'] = min(end_app, end_main)
    return date_ranges


def load_activations(activations_file):
    activation_dtype = {
        'dataset': str,
        'appliance': str,
        'building': int,
        'start': str,
        'end': str
    }
    activation_names = ["dataset", "appliance", "building", "start", "end"]
    activation_parse_dates = ['start', 'end']
    activations = pd.read_csv(activations_file,
                              quotechar="'",
                              header=1,
                              encoding='utf-8',
                              names=activation_names,
                              dtype=activation_dtype,
                              parse_dates=activation_parse_dates)
    return activations


def generate_activations(datasets):
    columns = ['dataset', 'appliance', 'building', 'start', 'end']
    i = 0
    activations = pd.DataFrame(columns=columns)
    for appliance in appliances:
        for dataset in datasets:
            data = datasets[dataset]
            for building in [x + 1 for x in range(len(data.buildings))]:
                try:
                    data.set_window(
                        start=str(date_ranges[dataset][str(building)][appliance]['start']),
                        end=str(date_ranges[dataset][str(building)][appliance]['end'])
                    )
                    appliance_elec = data.buildings[building].elec[appliance]
                    get_act = appliance_elec.get_activations()
                    for a in get_act:
                        start = a.first_valid_index()
                        end = a.last_valid_index()
                        df = pd.DataFrame([[dataset, appliance, building, str(start), str(end)]], columns=columns)
                        activations = activations.append(df, ignore_index=True)
                        print("[{}][{}][{}][{}][{}][{}]".format(i, appliance, dataset, building, str(start), str(end)))
                        i = i + 1
                except TypeError:
                    continue
                except KeyError:
                    continue
    return activations


def get_random_activation(data, excludes=[]):
    while True:
        try:
            select = np.round(
                len(data.index - 1) * random.random()).astype(int)
            if select not in excludes:
                return select, data.loc[[select]].values[0]
        except KeyError as e:
            print("KeyError: ", e)
            traceback.print_exc()
            continue


def get_random_window_nonactivation_apply(row, window_size, windows_max_offset, start_range, end_range):
    row['start'], row['end'] = get_random_window_nonactivation(row['start'], row['end'], window_size,
                                                               windows_max_offset,
                                                               start_range, end_range)
    return row


def get_random_window_nonactivation(start_act, end_act, window_size, windows_max_offset, start_range, end_range):
    k = 0
    while True:
        offset = int(np.round(windows_max_offset * random.random()))
        start_win = start_act + dt.timedelta(seconds=offset)
        end_win = start_win + dt.timedelta(seconds=window_size)
        start_win, end_win = fix_start_end_range(start_win, end_win, start_range, end_range)
        if start_act <= start_win and end_act >= end_win:
            return start_win, end_win
        k = k + 1
        if k > 1000:
            print("--------> get_random_window_activation: k > 1000 iterations")
            return np.NaN, np.NaN


def get_random_window_activation_apply(row, window_size, windows_max_offset, start_range, end_range):
    row['start'], row['end'] = get_random_window_activation(row['start'], row['end'], window_size, windows_max_offset,
                                                            start_range, end_range)
    return row


def get_random_window_activation(start_act, end_act, window_size, windows_max_offset, start_range, end_range):
    k = 0
    while True:
        offset = int(np.round(windows_max_offset * ((random.random() - 0.5) * 2)))
        start_win = start_act + dt.timedelta(seconds=offset)
        end_win = start_win + dt.timedelta(seconds=window_size)
        start_win, end_win = fix_start_end_range(start_win, end_win, start_range, end_range)
        if (start_act <= start_win and end_act >= end_win) or \
                (abs((pd.Timestamp(start_win) - pd.Timestamp(start_act)).total_seconds()) <= windows_max_offset) or \
                (abs((pd.Timestamp(end_win) - pd.Timestamp(end_act)).total_seconds()) <= windows_max_offset):
            return start_win, end_win
        k = k + 1
        if k > 1000:
            print("--------> get_random_window_activation: k > 1000 iterations")
            return np.NaN, np.NaN


def get_random_building_from_act(data_act_app, appliance, buildings):
    while True:
        building = str(random.choice(buildings))
        d = data_act_app[data_act_app['appliance'] == appliance]['building']
        for k in d:
            if k == building:
                return int(building)


def fix_start_end_range(start_win, end_win, start_range, end_range):
    k = 0
    while True:
        if end_win > end_range:
            diff = np.round((end_win - end_range).total_seconds()).astype(int)
            start_win = start_win - dt.timedelta(seconds=diff)
            end_win = end_win - dt.timedelta(seconds=diff)
        if start_win < start_range:
            diff = np.round((start_range - start_win).total_seconds()).astype(int)
            start_win = start_win + dt.timedelta(seconds=diff)
            end_win = end_win + dt.timedelta(seconds=diff)
        if end_win <= end_range and start_win >= start_range:
            start_win = np.datetime64("{}".format(start_win))
            end_win = np.datetime64("{}".format(end_win))
            return start_win, end_win
        k = k + 1
        if k > 1000:
            print("--------> fix_start_end_range: k > 1000 iterations")
            return np.NaN, np.NaN


def random_datetime(start, end):
    if type(start) is pd.Timestamp and type(end) is pd.Timestamp:
        return start + pd.Timedelta(round(random.random() * (end - start).total_seconds(), 0), unit='s')
    else:
        return start + (random.random() * (end.total_seconds() - start.total_seconds()))


def win_not_intersect_activations(start_win, end_win, activations):
    #if not activations.empty and not activations.query('('
    #                                                   '    start < @start_win < end | '
    #                                                   '    start < @end_win < end | '
    #                                                  '    (@start_win < start & @end_win > end)'
    #                                                   ')').empty:
    if activations.empty:
        return True

    for i, act in activations.iterrows():
        if (act['start'] < start_win < act['end']) or \
           (act['start'] < end_win < act['end']) or \
           (start_win < act['start'] and end_win > act['end']):
            return False
        else:
            if end_win < act['start']:
                return True
    return True


def has_act_set(row, activations):
    if win_not_intersect_activations(row['start'], row['end'], activations):
        row['has_act'] = 0
    else:
        row['has_act'] = 1
    return row


def win_check_valid_size(row, win_size):
    duration_act = (row['end'] - row['start']).total_seconds()
    duration_act = np.round(duration_act)
    duration_act = int(duration_act)
    if win_size <= duration_act:
        return True
    else:
        return False


def noise_window(row, app, delta, win_size):
    start_range = date_ranges[row['dataset']][str(int(row['building']))][app]['start']
    end_range = date_ranges[row['dataset']][str(int(row['building']))][app]['end']
    start = row['start']
    start = random_datetime(start - delta, start + delta)
    end = start + pd.Timedelta(win_size, unit='s')
    start, end = fix_start_end_range(start, end, start_range, end_range)
    syn_row = [row['dataset'], row['building'], row['has_act'], start, end]
    return syn_row


def cast_int(x):
    return int(x)


def check_gap(row, elec_app_series, elec_main_series):
    ret = get_x_y_data(row['start'], row['end'], elec_main_series, elec_app_series)
    row['gap'] = ret is None
    return row


def get_time_series_raw(dataset, building, app):
    elec_app_series = pd.Series.from_csv('{}_{}_{}_raw.csv'.format(dataset, int(building), app), index_col=0,
                                         parse_dates=True, header=0, infer_datetime_format=True)
    elec_main_series = pd.Series.from_csv('{}_{}_{}_main_raw.csv'.format(dataset, int(building), app), index_col=0,
                                          parse_dates=True, header=0, infer_datetime_format=True)
    return elec_app_series, elec_main_series


def get_time_series(dataset, building, app):
    elec_app_series = pd.Series.from_csv('{}_{}_{}.csv'.format(dataset, int(building), app), index_col=0,
                                         parse_dates=True, header=0, infer_datetime_format=True)
    elec_main_series = pd.Series.from_csv('{}_{}_{}_main.csv'.format(dataset, int(building), app), index_col=0,
                                          parse_dates=True, header=0, infer_datetime_format=True)
    return elec_app_series, elec_main_series


def generate_windows(date_ranges, app, data_act, win_size, output_win):
    # obtengo la cantidad de activaciones para ese electrodomestico
    win_build = 10000
    win_max_offset = windows_max_offset[app]
    win_df_columns = ["dataset", "building", "has_act", "start", "end"]
    win_df = pd.DataFrame(columns=win_df_columns)
    win_df_noact = pd.DataFrame(columns=win_df_columns)

    no_validate_gaps = {('ukdale', 1.0): 1}
    # no_validate_gaps = {}

    # generar ventanas con activaciones

    #while (len(win_df) < 100000):
    datasets = data_act['dataset'].unique()
    for dataset in datasets:
        date_ranges_filter = date_ranges[date_ranges['dataset'] == dataset]
        data_act_dataset_query = data_act[data_act['dataset'] == dataset]
        buildings = data_act_dataset_query['building'].unique()
        for building in buildings:
            print('    -> ({},{}) processing'.format(dataset, building))
            data_act_query = data_act_dataset_query[data_act_dataset_query['building'] == building]
            if data_act_query.empty:
                print('    -> ({},{}) no activations for this building '.format(dataset, building))
            else:
                print('    -> ({},{}) get time series '.format(dataset, building))
                data_act_filtered = data_act_query.copy()
                validate_gaps = True
                if (dataset, building) not in no_validate_gaps:
                    try:
                        elec_app_series, elec_main_series = get_time_series_raw(dataset, building, app)
                    except ValueError:
                        del elec_main_series
                        del elec_app_series
                        print("[WARN] Error al obtener las series. No se validaran los gaps")
                        validate_gaps = False
                        no_validate_gaps[(dataset, building)] = 1
                else:
                    validate_gaps = False
                dates = date_ranges_filter[date_ranges_filter['building'] == building]
                start_range = dates['start'].iloc[0]
                end_range = dates['end'].iloc[0]

                # generar con activaciones BEGIN
                print('    -> ({},{}) generate windows with activations '.format(dataset, building))
                data_act_filtered['has_act'] = 1
                data_act_filtered = data_act_filtered.sample(win_build, replace=True).reset_index()
                data_act_filtered = apply_by_multiprocessing(data_act_filtered, get_random_window_activation_apply,
                                                             args=(win_size, win_max_offset, start_range, end_range)
                                                             , axis=1, workers=cores)
                data_act_filtered = data_act_filtered.dropna(axis=0)
                print('    -> ({},{}) validate if windows has activations'.format(dataset, building))
                data_act_filtered = apply_by_multiprocessing(data_act_filtered, has_act_set, args=(data_act_query,),
                                                             axis=1, workers=cores)
                discards = len(data_act_filtered[data_act_filtered['has_act'] == 0])
                data_act_filtered = data_act_filtered[data_act_filtered['has_act'] == 1]

                if validate_gaps:
                    print('    -> ({},{}) validate gaps '.format(dataset, building))
                    data_act_filtered = apply_by_multiprocessing(data_act_filtered, check_gap, axis=1,
                                                                 workers=cores,
                                                                 args=(elec_app_series, elec_main_series))
                    gaps = len(data_act_filtered[data_act_filtered['gap'] == True])
                    data_act_filtered = data_act_filtered[data_act_filtered['gap'] == False]
                else:
                    gaps = np.NaN

                data_act_filtered = data_act_filtered[['dataset', 'building', 'has_act', 'start', 'end']]
                print('    -> ({},{}) append windows with to win_df'.format(dataset, building))
                win_df = win_df.append(data_act_filtered, ignore_index=True)
                print("- Generar ventanas con activaciones para [{}][{}][{}] -> {}, error: {}, gaps: {}".
                      format(dataset, building, app, len(data_act_filtered), discards, gaps))

                # generar con activaciones END

                # generar sin activaciones BEGIN
                print('    -> ({},{}) generate windows without activations '.format(dataset, building))
                del data_act_filtered
                data_act_filtered = data_act_query.copy()
                data_act_filtered['has_act'] = 0
                first = None
                if data_act_filtered['start'].iloc[0] > start_range:
                    # agrego el primer intervalo sin activaciones
                    first = pd.DataFrame([[dataset, building, 0, start_range, data_act_filtered['start'].iloc[0]]],
                                         columns=win_df_columns)
                data_act_filtered['start'] = data_act_filtered['end']
                data_act_filtered['end'] = data_act_filtered['start'].shift(-1).fillna(end_range)
                if first is not None:
                    data_act_filtered = data_act_filtered.append(first, ignore_index=True)

                data_act_filtered = data_act_filtered.sample(win_build, replace=True).reset_index()
                data_act_filtered = apply_by_multiprocessing(data_act_filtered,
                                                             get_random_window_nonactivation_apply,
                                                             args=(win_size, win_max_offset, start_range, end_range),
                                                             axis=1, workers=cores)
                data_act_filtered = data_act_filtered.dropna(axis=0)
                print('    -> ({},{}) validate if windows has no activations '.format(dataset, building))
                data_act_filtered = apply_by_multiprocessing(data_act_filtered, has_act_set, args=(data_act_query,),
                                                             axis=1, workers=cores)
                discards = len(data_act_filtered[data_act_filtered['has_act'] == 1])
                data_act_filtered = data_act_filtered[data_act_filtered['has_act'] == 0]

                if validate_gaps:
                    print('    -> ({},{}) validate gaps '.format(dataset, building))
                    data_act_filtered = apply_by_multiprocessing(data_act_filtered, check_gap, axis=1,
                                                                 workers=cores,
                                                                 args=(elec_app_series, elec_main_series))
                    gaps = len(data_act_filtered[data_act_filtered['gap'] == True])
                    data_act_filtered = data_act_filtered[data_act_filtered['gap'] == False]
                else:
                    gaps = np.NaN

                data_act_filtered = data_act_filtered[['dataset', 'building', 'has_act', 'start', 'end']]

                print('    -> ({},{}) append windows with to win_df'.format(dataset, building))

                win_df_noact = win_df_noact.append(data_act_filtered, ignore_index=True)
                print("- Generar ventanas sin activaciones para [{}][{}][{}] -> {}, error: {}, gaps: {}".
                      format(dataset, building, app, len(data_act_filtered), discards, gaps))
                # generar sin activaciones END
                win_df = win_df.append(win_df_noact, ignore_index=True)

    print("- Ordenar por [dataset,building,start]")
    win_df = win_df.sort_values(['dataset', 'building', 'start'], ascending=[1, 1, 1])
    print("- Normalizar building a int")
    win_df['building'] = apply_by_multiprocessing(win_df['building'], cast_int, workers=cores)
    print("- Normalizar has_act a int")
    win_df['has_act'] = apply_by_multiprocessing(win_df['has_act'], cast_int, workers=cores)
    print("- Guardar win_df a csv")
    win_df.to_csv(path_or_buf=output_win, columns=win_df_columns, quotechar="'", encoding='utf-8', index=False)
    print("- Guardado win_df")


def get_x_y_data_row(row, elec_main_series, elec_app_series):
    row['building'] = int(row['building'])
    ret = get_x_y_data(row['start'], row['end'], elec_main_series, elec_app_series)
    if ret is not None:
        return pd.Series(row.tolist() + ret.tolist())
    else:
        return pd.Series(np.NaN)


def get_x_y_data(start_win, end_win, elec_main_series, elec_app_series):
    try:
        elec_app_series_win = elec_app_series[start_win:end_win]
        elec_main_series_win = elec_main_series[start_win:end_win]
        if not elec_app_series_win.empty and not elec_main_series_win.empty:
            x = elec_main_series_win.values
            y = elec_app_series_win.values
            if not np.isnan(x).any() and not np.isnan(y).any():
                row = np.append(x, y)
                return row
            else:
                # print("- get_X_Y({},{}) nan".format(start_win, end_win))
                return None
        else:
            # print("- get_X_Y({},{}) empty".format(start_win, end_win))
            return None
    except:
        return None


def generate_windows_data(app, output_data):
    windows_dtype = {
        'dataset': str,
        'building': int,
        'hash_act': int,
        'start': str,
        'end': str
    }
    windows_names = ['dataset', 'building', 'has_act', 'start', 'end']
    windows_parse_dates = ['start', 'end']
    windows_data = pd.read_csv(windows_df_files[app], quotechar="'", header=1, encoding='utf-8', names=windows_names,
                               dtype=windows_dtype, parse_dates=windows_parse_dates, date_parser=np.datetime64)
    file = open(output_data, "a")
    for d in windows_data['dataset'].unique():
        for b in windows_data[windows_data['dataset'] == d]['building'].unique():
            try:
                windows_data_filtered = windows_data.query('dataset == @d & building == @b')
                elec_app_series, elec_main_series = get_time_series(d, b, app)
                windows_data_filtered = apply_by_multiprocessing(windows_data_filtered, get_x_y_data_row, axis=1,
                                                                 args=(elec_main_series, elec_app_series),
                                                                 workers=cores)
                windows_data_filtered.to_csv(path_or_buf=file, quotechar="'", encoding='utf-8', index=False,
                                             header=False)
            except ValueError:
                continue
    file.close()


if __name__ == '__main__':

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "a:w:f:ow:od:",
                               ["prefix=", "app=", "window=", "freq=", "output_win=", "output_data="])
    app = None
    window = None
    frq = None
    work_directory = ""
    output_data = "win_data.csv"
    output_win = "win.csv"
    for opt, arg in opts:
        if opt == "--prefix":
            work_directory = arg
        if opt in ("-a", "--app"):
            app = arg
        if opt in ("-w", "--window"):
            window = int(arg)
        if opt in ("-f", "--freq"):
            freq = int(arg)
        if opt in ("-ow", "--output_win"):
            output_win = arg
        if opt in ("-od", "--output_data"):
            output_data = arg

    output_win = work_directory + "/" + output_win
    output_data = work_directory + "/" + output_data

    start_time = time.time()
    # if not exist date_ranges_csv
    if os.path.isfile(date_ranges_csv):
        date_ranges = load_date_ranges(app)
        date_ranges = date_ranges[date_ranges['appliance'] == app]
    else:
        exit()

    print("date_ranges: {} seconds".format(time.time() - start_time))
    start_time = time.time()

    # if not exist activations file data, generate it
    if os.path.exists(activations_csv):
        data_activations = load_activations(activations_csv)
    else:
        exit()
    print("load_activations: {} seconds".format(time.time() - start_time))
    start_time = time.time()
    data_activations = data_activations[data_activations['appliance'] == app]
    data_activations = data_activations.sort_values(['appliance', 'dataset', 'building', 'start'],
                                                    ascending=[1, 1, 1, 1]).reset_index()

    if not os.path.exists(output_win):
        generate_windows(date_ranges, app, data_activations, window, output_win)

    start_time = time.time()
    if not os.path.exists(output_data):
        generate_windows_data(app, output_data)


