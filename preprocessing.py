import os

import labelbox
import mne
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq


class EEGPreprocessing:
    def __init__(self, data_dir: str):
        # TODO:
        #  1. Сделать универсальный метод с выбором из указанного проекта labelbox
        #  2. Связать метки labelbox с выбранным файлом пользователя

        LB_API_KEY = ('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHBsaWg4eW8wNWpyMDd3YTJ1N3ZoejQzIiwib3JnYW'
                      '5pemF0aW9uSWQiOiJjbHBsaWg4eHkwNWpxMDd3YTRwNmhneDd2IiwiYXBpS2V5SWQiOiJjbHBsbWgxaTMwOHExMDd1M2d5Z'
                      'ThnbG5qIiwic2VjcmV0IjoiNDMxMzA0YTJjYWVlYThmZTg3Yzk2YTAwNTI0OTE1MDciLCJpYXQiOjE3MDEzNzQ1NDAsImV4'
                      'cCI6MjMzMjUyNjU0MH0.7i65i3kaBGzU1XXIfm75u_Ghm3EL8sB89wmGlpo_VLc')
        PROJECT_ID = 'clplk9kff02sp07vohdro0a53'
        client = labelbox.Client(api_key=LB_API_KEY)
        project = client.get_project(PROJECT_ID)
        labels = project.export_v2(params={
            "data_row_details": True,
            "metadata_fields": True,
            "attachments": True,
            "project_details": True,
            "performance_details": True,
            "label_details": True,
            "interpolated_frames": True
        })
        labels.wait_till_done()
        self.filt_raw = dict()
        self.export_json = labels.result
        self.rate = int(2199 / (1 * 60 + 13.299))
        self.sfreq = 250.0
        self.filenames = [f'{data_dir}/Артемий правая нога.txt',
                          f'{data_dir}/Артемий правая рука.txt',
                          f'{data_dir}/Артемий рука расслаблена.txt',
                          f'{data_dir}/Владимир правая нога.txt',
                          f'{data_dir}/Владимир правая рука ещё раз.txt',
                          f'{data_dir}/Владимир правая рука.txt',
                          f'{data_dir}/Владимир только воображаемые движения рука.txt',
                          f'{data_dir}/Сергей правая рука на расслаблении.txt']

    def create_init_dataset(self, dir_path: str, freq_amount: int = 150):
        files = os.listdir(dir_path)

        cols = ['idx', 'name', 'real', 'action', 'body_part', 'num_epoch']
        metrics_names = ['max', 'mean', 'q25', 'median', 'q75']
        canals = ['Fp1', 'Fp2', 'AF3', 'AF4', 'P1', 'P2']
        for j in range(len(canals)):
            for i in range(len(metrics_names)):
                cols.append(canals[j] + '_' + metrics_names[i])
            for i in range(freq_amount):
                cols.append(canals[j] + '_' + str(i))
        ans = pd.DataFrame([], columns=cols)
        dct_list_1 = []
        dct_list_2 = []
        idx = 0
        for filename in files:
            filepath = f'{dir_path}/{filename}'
            if filepath not in self.filenames:
                continue
            filename_idx = self.filenames.index(filepath)

            filt_raw_save_1, filt_raw_save_2 = self.create_initial_data(filename_idx)
            self.filt_raw[filename_idx] = (filt_raw_save_1, filt_raw_save_2)
            data_1 = pd.DataFrame(filt_raw_save_1.get_data().T,
                                  columns=['Fp1', 'Fp2', 'AF3', 'AF4', 'P1', 'P2', 'event'])
            data_2 = pd.DataFrame(filt_raw_save_2.get_data().T,
                                  columns=['Fp1', 'Fp2', 'AF3', 'AF4', 'P1', 'P2', 'event'])
            metrics_1 = self.create_metrics(data_1)
            metrics_2 = self.create_metrics(data_2)
            dct_list_1.append([])
            dct_list_2.append([])
            for i in range(metrics_1.shape[0]):
                data_11 = data_1.iloc[int(metrics_1['begin'][i]):int(metrics_1['end'][i]) + 1, :].copy()
                data_11.reset_index(inplace=False)
                body_part = 'arm' if 'рука' in self.filenames[filename_idx] else 'leg'
                row = [idx, filename_idx, 1, metrics_1['type'][i], body_part, i]
                for canal_idx in range(len(canals)):
                    temp = []
                    dct_1 = self.create_freq(data_11, canals[canal_idx])
                    val_lst = list(dct_1.values())
                    temp.append(max(val_lst))
                    temp.append(max(val_lst) / len(val_lst))
                    temp.append(val_lst[int(len(val_lst) * 0.25)])
                    temp.append(val_lst[int(len(val_lst) * 0.5)])
                    temp.append(val_lst[int(len(val_lst) * 0.75)])
                    for k in range(len(list(dct_1.keys()))):
                        temp.append(dct_1[list(dct_1.keys())[k]])
                    if len(temp) > freq_amount + len(metrics_names):
                        temp = temp[:freq_amount + len(metrics_names) + 1]
                    while len(temp) < freq_amount + len(metrics_names):
                        temp.append(0)
                    for item in temp:
                        row.append(item)
                ans.loc[ans.shape[0]] = row
                idx += 1

            for i in range(metrics_2.shape[0]):
                data_22 = data_2.iloc[int(metrics_2['begin'][i]):int(metrics_2['end'][i]) + 1, :].copy()
                data_22.reset_index(inplace=False)
                body_part = 'arm' if 'рука' in self.filenames[filename_idx] else 'leg'
                row = [idx, filename_idx, 0, metrics_2['type'][i], body_part, i]
                for canal_idx in range(len(canals)):
                    temp = []
                    dct_2 = self.create_freq(data_22, canals[canal_idx])
                    val_lst = list(dct_2.values())
                    temp.append(max(val_lst))
                    temp.append(max(val_lst) / len(val_lst))
                    temp.append(val_lst[int(len(val_lst) * 0.25)])
                    temp.append(val_lst[int(len(val_lst) * 0.5)])
                    temp.append(val_lst[int(len(val_lst) * 0.75)])
                    for k in range(len(list(dct_2.keys()))):
                        temp.append(dct_2[list(dct_2.keys())[k]])
                    if len(temp) > freq_amount + len(metrics_names):
                        temp = temp[:freq_amount + len(metrics_names) + 1]
                    while len(temp) < freq_amount + len(metrics_names):
                        temp.append(0)
                    for item in temp:
                        row.append(item)
                ans.loc[ans.shape[0]] = row
                idx += 1

        clear_ans = self.clear_cols(ans)
        return clear_ans

    def create_initial_data(self, filename_idx: int):
        """Преобразование одного измерения в датасет по индексу названия"""
        names = []
        time = []
        if len(self.export_json[filename_idx]['projects']['clplk9kff02sp07vohdro0a53']['labels']) > 0:
            for e in self.export_json[filename_idx]['projects']['clplk9kff02sp07vohdro0a53']['labels'][0]['annotations']\
                    ['frames'].keys():
                time.append(e)
                names.append(self.export_json[filename_idx]['projects']['clplk9kff02sp07vohdro0a53']['labels'][0]['annotations'][
                                 'frames'][e]['classifications'][0]['radio_answer']['name'])
        df = pd.read_csv(self.filenames[filename_idx], sep=',',
                         names=['Sample Index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3',
                                'EXG Channel 4',
                                'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7',
                                'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2',
                                'Other 0', 'Other 1', 'Other 2', 'Other 3', 'Other 4', 'Other 5', 'Other 6',
                                'Analog Channel 0', ' Analog Channel 1', ' Analog Channel 2',
                                'Timestamp', 'Other 7', 'Timestamp (Formatted)'], skiprows=5)
        tags = pd.DataFrame({'Frames': time, 'Type': names})
        tags['time'] = tags['Frames'].astype(int) / self.rate
        tags = tags.sort_values(by='time', ascending=True).reset_index(drop=True)
        minn = 10000
        for i in range(1, tags.shape[0]):
            if tags['time'][i] - tags['time'][i - 1] < minn:
                minn = tags['time'][i] - tags['time'][i - 1]
        tags['idx'] = (tags['time'].astype(float) * self.sfreq).astype(int)
        df["event"] = np.nan
        n = 0
        for i, d in enumerate(df['event']):

            if i == tags['idx'][n] and n < len(tags['idx']) - 1:
                df['event'][i] = tags['Type'][n]
                n += 1
            else:
                df['event'][i] = None
        df1 = df.iloc[
              df[df['event'] == '2'].index.values.astype(int)[0]:df[df['event'] == '2'].index.values.astype(int)[1]]
        df2 = df.iloc[df[df['event'] == '3'].index.values.astype(int)[0]:]

        data1 = df1.loc[:,
                ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5',
                 'event']]
        data1.rename(columns={' EXG Channel 0 ': 'Fp1', 'EXG Channel 1 ': 'Fp2', 'EXG Channel 2 ': 'AF3',
                              'EXG Channel 3 ': 'AF4', 'EXG Channel 4 ': 'P1', 'EXG Channel 5 ': 'P2'}, inplace=True)
        data2 = df2.loc[:,
                ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5',
                 'event']]
        data2.rename(columns={' EXG Channel 0 ': 'Fp1', 'EXG Channel 1 ': 'Fp2', 'EXG Channel 2 ': 'AF3',
                              'EXG Channel 3 ': 'AF4', 'EXG Channel 4 ': 'P1', 'EXG Channel 5 ': 'P2'}, inplace=True)

        ch_names = ['Fp1', 'Fp2', 'AF3', 'AF4', 'P1', 'P2', 'event']
        ch_types = ['eeg'] * 6 + [
            'stim']  # задаем тип каналов: триггер-канал (содержит информацию о стимулах) и 6 каналов ЭЭГ
        # создаем структуру с метаданными
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=self.sfreq);
        raw1 = mne.io.RawArray(data1.transpose(), info, verbose=True)
        raw2 = mne.io.RawArray(data2.transpose(), info, verbose=True)
        montage = mne.channels.make_standard_montage(
            'standard_1020')  # расположение электродов в соответствии с международной системой 10-20
        raw1.set_montage(montage);
        montage = mne.channels.make_standard_montage(
            'standard_1020')  # расположение электродов в соответствии с международной системой 10-20
        raw2.set_montage(montage);
        raw1.interpolate_bads(reset_bads=True, mode='accurate')
        mne.set_eeg_reference(raw1);
        raw1.filter(1, 45, fir_design='firwin', skip_by_annotation='edge')
        raw2.interpolate_bads(reset_bads=True, mode='accurate')
        mne.set_eeg_reference(raw2);
        raw2.filter(1, 45, fir_design='firwin', skip_by_annotation='edge')
        filt_raw1 = raw1.copy()
        filt_raw_save_1 = filt_raw1.load_data().filter(l_freq=1., h_freq=45.)
        filt_raw2 = raw2.copy()
        filt_raw_save_2 = filt_raw2.load_data().filter(l_freq=1., h_freq=45.)
        return filt_raw_save_1, filt_raw_save_2

    @staticmethod
    def create_metrics(data: pd.DataFrame):
        tags_1 = data[data['event'] >= 0].copy()
        metrics_1 = pd.DataFrame([], columns=['type', 'begin', 'end'])
        for j in range(1, len(tags_1.index)):
            i_prev = tags_1.index[j - 1]
            i = tags_1.index[j]
            if tags_1['event'][i] == tags_1['event'][i_prev]:
                metrics_1.loc[metrics_1.shape[0]] = [tags_1['event'][i], tags_1.index[j - 1], tags_1.index[j]]
        return metrics_1

    @staticmethod
    def create_freq(df1, col1):
        N = df1.shape[0]
        yf = rfft(df1[col1].values)
        xf = rfftfreq(N, 1 / 250)
        dct = {}
        for x, y in zip(xf, np.abs(yf)):
            if x >= 4 and x <= 45:
                dct[x] = y
        return dct

    @staticmethod
    def clear_cols(df, init_cols=('name', 'real', 'action', 'body_part', 'num_epoch'), prop=0.9):
        a = df.astype(bool).mean(axis=0).to_dict()
        new_col = list(init_cols)
        for item in a.keys():
            if a[item] > prop and item not in new_col:
                new_col.append(item)
        return df[new_col]

    @staticmethod
    def create_hist(df_all, df_new, channel, metric, ax):
        mean_values = list(df_all[channel + '_' + metric])
        dot = df_new[channel + '_' + metric].mean()
        ax.hist(mean_values, label='Распределение значений');
        ax.axvline(x=dot, color='red', label='Значение по исследованию')
        ax.legend()
        ax.set_title('Канал ' + channel + ', метрика ' + metric)
        ax.set_xlabel('Мощность гармоники')
        ax.set_ylabel('Число гармоник')

    @staticmethod
    def create_fourier(df_all, df_new, channel, max_freq, ax):
        metric_cols = []
        pre_metrics_names = ['max', 'mean', 'q25', 'median', 'q75']
        metrics_names = []
        for i in range(len(pre_metrics_names)):
            metrics_names.append(channel + '_' + pre_metrics_names[i])
        for col in df_all.columns:
            if channel in col and col not in metrics_names:
                metric_cols.append(col)
        all_values = []
        current_values = []
        labels = []
        for col in metric_cols:
            all_values.append(round(df_all[col].mean()))
            current_values.append(round(df_new[col][0]))
            labels.append(col[len(channel):])
        pd.DataFrame({'Среднее значение': all_values[:max_freq],
                      'Значение по исследованию': current_values[:max_freq]},
                     index=labels[:max_freq]).plot.bar(figsize=(15, 10),
                                                       xlabel='Частота гармоники, канал ' + channel,
                                                       ylabel='Мощность гармоники', title=str('Канал ' + channel),
                                                       ax=ax);
