import datetime as dt
import traceback
import pandas as pd
import numpy as np
import random
import time
import sys
from nilmtk import DataSet
import os
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

dataset_ukdale_path = "/home/bruno/Escritorio/Data/ukdale.h5"
dataset_redd_path = "/home/bruno/Escritorio/redd.h5"
dataset_iawe_path = "/home/bruno/Escritorio/Data/iawe.h5"
# dataset_greend_path = "/home/bruno/Escritorio/Data/greend.h5"
dataset_combed_path = "/home/bruno/Escritorio/Data/combed.h5"

# number windows to extract foreach dataset
k = 1000

appliances = ['dish washer', 'fridge', 'kettle', 'microwave', 'washing machine']

activations_csv = "/home/bruno/Escritorio/Data/activations.csv"
windows_csv = "/home/bruno/Escritorio/Data/windows_data.csv"
date_ranges_csv = "/home/bruno/Escritorio/Data/date_ranges.csv"

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

windows_period = {
    'dish washer': 120,
    'fridge': 30,
    'kettle': 30,
    'microwave': 30,
    'washing machine': 60
}

windows_df_files = {
    'dish washer': '/home/bruno/Escritorio/Data/win_dish_washer.csv',
    'fridge': '/home/bruno/Escritorio/Data/win_fridge.csv',
    'kettle': '/home/bruno/Escritorio/Data/win_kettle.csv',
    'microwave': '/home/bruno/Escritorio/Data/win_microwave.csv',
    'washing machine': '/home/bruno/Escritorio/Data/win_washing_machine.csv'
}

windows_df_data_files = {
    'dish washer': '/home/bruno/Escritorio/Data/windat_dish_washer.csv',
    'fridge': '/home/bruno/Escritorio/Data/windat_fridge.csv',
    'kettle': '/home/bruno/Escritorio/Data/windat_kettle.csv',
    'microwave': '/home/bruno/Escritorio/Data/windat_microwave.csv',
    'washing machine': '/home/bruno/Escritorio/Data/windat_washing_machine.csv'
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


def load_date_ranges(datasets, appliances):
    df = pd.read_csv(date_ranges_csv, sep=',', header=0, encoding='utf-8',
                     names=['dataset', 'building', 'appliance', 'start', 'end'],
                     date_parser=pd.to_datetime,
                     parse_dates=['start', 'end'])
    date_ranges = {}
    for dataset in datasets:
        date_ranges[dataset] = {}
        for b in [x + 1 for x in range(len(datasets[dataset].buildings))]:
            date_ranges[dataset][str(b)] = {}
            for app in appliances:
                date_ranges[dataset][str(b)][app] = {}

    for _, d in df.iterrows():
        date_ranges[d['dataset']][str(d['building'])][d['appliance']]['start'] = d['start']
        date_ranges[d['dataset']][str(d['building'])][d['appliance']]['end'] = d['end']
    return date_ranges


def generate_date_ranges(datasets, appliances):
    date_ranges = {}
    for dataset in datasets:
        date_ranges[dataset] = {}
        for b in [x + 1 for x in range(len(datasets[dataset].buildings))]:
            date_ranges[dataset][str(b)] = {}
            df_main = datasets[dataset].buildings[b].elec.mains().power_series().next()
            start_main = df_main.first_valid_index()
            end_main = df_main.last_valid_index()
            df_main = None
            for app in appliances:
                try:
                    df_app = datasets[dataset].buildings[b].elec[app].power_series().next()
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
                        print "[{}][{}][{}][{}][{}][{}]".format(i, appliance, dataset, building, str(start), str(end))
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
            print "KeyError: ", e
            traceback.print_exc()
            continue


def get_random_window_activation(activation, window_size, start_range, end_range):
    duration_act = (activation['end'] - activation['start']).total_seconds()
    duration_act = np.round(duration_act)
    duration_act = int(duration_act)
    offset = abs(window_size - duration_act)
    offset = int(np.round(offset * random.random()))
    start_win = activation['start'] - dt.timedelta(seconds=offset)
    end_win = start_win + dt.timedelta(seconds=window_size)
    return fix_start_end_range(start_win, end_win, start_range, end_range)


def get_random_building_from_act(data_act_app, appliance, buildings):
    while True:
        building = str(random.choice(buildings))
        d = data_act_app[data_act_app['appliance'] == appliance]['building']
        for k in d:
            if k == building:
                return int(building)


def fix_start_end_range(start_win, end_win, start_range, end_range):
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


def random_datetime(start, end):
    if type(start) is pd.Timestamp and type(end) is pd.Timestamp:
        return start + pd.Timedelta(round(random.random() * (end - start).total_seconds(), 0), unit='s')
    else:
        return start + (random.random() * (end.total_seconds() - start.total_seconds()))


def win_not_intersect_activations(dataset, building, start_win, end_win, activations, win_df):
    try:
        if not activations.empty and not activations.query('dataset == @dataset & building == @building & '
                                                           '('
                                                           '    start < @start_win < end | '
                                                           '    start < @end_win < end | '
                                                           '    (@start_win < start & @end_win > end)'
                                                           ')').empty:
            return False
        if not win_df.empty and not win_df.query('dataset == @dataset & building == @building & ('
                                                 '  start < @start_win < end | '
                                                 '  start < @end_win < end |'
                                                 '  (@start_win < start & @end_win > end)'
                                                 ')').empty:
            return False
        return True
    except KeyError as e:
        print e
    except ValueError as e:
        print e


def generate_windows(datasets, date_ranges, app, data_act):
    # obtengo la cantidad de activaciones para ese electrodomestico
    k = 0
    win_size = windows_size[app]
    win_df_columns = ["dataset", "building", "has_act", "start", "end"]
    win_df = pd.DataFrame(columns=win_df_columns)
    win_df_noact = pd.DataFrame(columns=win_df_columns)

    # generar ventanas con activaciones
    for i, act in data_act.iterrows():
        try:
            start_range = date_ranges[act['dataset']][str(act['building'])][app]['start']
            end_range = date_ranges[act['dataset']][str(act['building'])][app]['end']
            start, end = get_random_window_activation(act, win_size, start_range, end_range)
            k = k + 1
            print "ACT: 1 -> [{}] {}, {}, {}, {}, {}".format(k, act['dataset'], act['building'], app, start, end)
            row = [act['dataset'], int(act['building']), 1, start, end]
            win_df = win_df.append(pd.DataFrame([row], columns=win_df_columns), ignore_index=True)
            if k > 10:
                break
        except KeyError:
            continue

    invalid_count = 0
    for i, act in win_df.iterrows():
        try:
            while True:
                data_act_filter = data_activations[data_activations["dataset"] == act['dataset']]
                building = act['building']
                data_act_filter = data_act_filter[data_act_filter["building"] == building]
                random_row = win_df.sample()
                offset = end_range - start_range
                offset = pd.Timedelta(round(random.random() * offset.total_seconds(), 0), unit='s')
                start_win = start_range + offset
                end_win = start_win + pd.Timedelta(win_size, unit='s')
                start_win, end_win = fix_start_end_range(start_win, end_win, start_range, end_range)
                if win_not_intersect_activations(act['dataset'], building, start_win, end_win, data_act_filter,
                                                 win_df) and \
                        win_not_intersect_activations(act['dataset'], building, start_win, end_win, data_act_filter,
                                                      win_df_noact):
                    row = [act['dataset'], int(building), 0, start_win, end_win]
                    win_df_noact = win_df_noact.append(pd.DataFrame([row], columns=win_df_columns), ignore_index=True)
                    print "ACT: 0 -> [{}] {}, {}, {}, {}, {}".format(i, act['dataset'], int(building), app, start_win,
                                                                     end_win)
                    i = i + 1
                    break
                else:
                    invalid_count = invalid_count + 1
                    if invalid_count > 100:
                        invalid_count = 0
                        break
        except KeyError:
            continue
    win_df = win_df.append(win_df_noact, ignore_index=True)

    delta = pd.Timedelta(round(0.1 * win_size, 0), unit='s')
    # while len(win_df) < 1000000:
    while len(win_df) < 0:
        for i, row in win_df.iterrows():
            start_range = date_ranges[row['dataset']][str(int(row['building']))][app]['start']
            end_range = date_ranges[row['dataset']][str(int(row['building']))][app]['end']
            start = row['start']
            start = random_datetime(start - delta, start + delta)
            end = start + pd.Timedelta(win_size, unit='s')
            start, end = fix_start_end_range(start, end, start_range, end_range)
            syn_row = [row['dataset'], row['building'], row['has_act'], start, end]
            win_df = win_df.append(pd.DataFrame([syn_row], columns=win_df_columns), ignore_index=True)
            print "SYN ACT: {} -> [{}] {}, {}, {}, {}, {}".format(row['has_act'], len(win_df),
                                                                  row['dataset'], int(row['building']), app, start, end)

    win_df = win_df.sort(columns=['dataset', 'building', 'start'], ascending=[1, 1, 1])
    win_df['building'] = win_df['building'].apply(lambda x: int(x))
    win_df['has_act'] = win_df['has_act'].apply(lambda x: int(x))
    win_df.to_csv(path_or_buf=windows_df_files[app],
                  columns=win_df_columns,
                  quotechar="'", encoding='utf-8', index=False)


def get_X_Y_data(start_win, end_win, elec_main_series, elec_app_series):
    try:
        elec_app_series_win = elec_app_series[start_win:end_win]
        elec_main_series_win = elec_main_series[start_win:end_win]
        if not elec_app_series_win.empty and not elec_main_series_win.empty:
            x = np.array([val for val in elec_main_series_win])
            y = np.array([val for val in elec_app_series_win])
            if np.isnan(x).any():
                print "x tiene nan: " + x
            if np.isnan(y).any():
                print "y tiene nan: " + y
            row = np.append(x, y)
            return row
        else:
            return None
    except:
        e = sys.exc_info()[0]
        print e
        return None


def generate_windows_data(app):
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
    data = {}
    data["ukdale"] = DataSet(dataset_ukdale_path)
    data["redd"] = DataSet(dataset_redd_path)
    file = open(windows_df_data_files[app], "a")
    for d in windows_data['dataset'].unique():
        for b in windows_data[windows_data['dataset'] == d]['building'].unique():
            try:
                windows_data_filtered = windows_data.query('dataset == @d & building == @b')
                buff = pd.DataFrame()
                elec = data[d].buildings[b].elec
                elec_app = elec[app]
                elec_main = elec.mains()
                elec_app_series = elec_app.power_series(ac_type='active', sample_period=windows_period[app]).next()
                elec_main_series = elec_main.power_series(ac_type='apparent', sample_period=windows_period[app]).next()
                for i, win in windows_data_filtered.iterrows():
                    row = get_X_Y_data(win['start'], win['end'], elec_main_series, elec_app_series)
                    if row is not None:
                        buff = buff.append([win.tolist() + row.tolist()], ignore_index=True)
                buff.to_csv(path_or_buf=file, quotechar="'", encoding='utf-8', index=False, header=False)
            except ValueError:
                continue
    file.close()


if __name__ == '__main__':
    start_time = time.time()
    datasets = load_datasets()
    print "load_datasets: {} seconds".format(time.time() - start_time)
    start_time = time.time()
    # if not exist date_ranges_csv
    if os.path.isfile(date_ranges_csv):
        date_ranges = load_date_ranges(datasets, appliances)
    else:
        date_ranges = generate_date_ranges(datasets, appliances)
        save_date_ranges(date_ranges, datasets, appliances)
    print "date_ranges: {} seconds".format(time.time() - start_time)
    start_time = time.time()

    # if not exist activations file data, generate it
    if os.path.exists(activations_csv):
        data_activations = load_activations(activations_csv)
    else:
        data_activations = generate_activations(datasets)
        columns = ['dataset', 'appliance', 'building', 'start', 'end']
        data_activations.to_csv(path_or_buf=activations_csv,
                                columns=columns, quotechar="'",
                                encoding='utf-8', index=False)
    print "load_activations: {} seconds".format(time.time() - start_time)
    start_time = time.time()

    data_activations = data_activations.sort(['appliance', 'dataset', 'building'], ascending=[1, 1, 1])

    app = "fridge"
    # for app in appliances:
    if not os.path.exists(windows_df_files[app]):
        generate_windows(datasets, date_ranges, app, data_activations[data_activations['appliance'] == app])

    exit()

    print "load_activations_windows: {} seconds".format(time.time() - start_time)

    start_time = time.time()
    # for app in appliances:
    app = "fridge"
    if not os.path.exists(windows_df_data_files[app]):
        generate_windows_data(app)

    print "load_activations_windows_data: {} seconds".format(time.time() - start_time)



