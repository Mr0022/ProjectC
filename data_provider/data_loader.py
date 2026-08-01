import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# Chronological train / val / test split -- single source of truth.
# Every dataset class in this file derives its borders from here, so changing
# the years below propagates to every model.
#
#     train : start of sample      ..  end of TRAIN_END_YEAR   (2012 - 2022)
#     val   : TRAIN_END_YEAR + 1   ..  end of VAL_END_YEAR     (2023 - 2024)
#     test  : VAL_END_YEAR + 1     ..  end of sample           (2025 - end)
# ==============================================================================
TRAIN_END_YEAR = 2022
VAL_END_YEAR = 2024


def year_split_borders(dates, seq_len, n_rows):
    """
    Row borders for the chronological split, indexed by set_type
    (0 = train, 1 = val, 2 = test).

    `dates` must be datetime-like and aligned 1:1 with the data rows.

    The val and test windows start seq_len rows early so the first forecast
    origin in each split has a complete look-back window. Those extra rows are
    context only -- they are never used as targets, so no target from an
    earlier split is scored twice.

    Returns (border1s, border2s); border2s[0] is also the train row count.
    """
    years = pd.Series(pd.to_datetime(pd.Series(dates).values)).dt.year
    train_end = int((years <= TRAIN_END_YEAR).sum())
    val_end = int((years <= VAL_END_YEAR).sum())
    border1s = [0, train_end - seq_len, val_end - seq_len]
    border2s = [train_end, val_end, n_rows]
    return border1s, border2s



class Dataset_Custom(Dataset):
    """
    Generic CSV dataset with a chronological year-based split.

    Two options exist for realized-variance work, both off by default so that
    every other dataset (ETT, Weather, Traffic, ...) is untouched:

      log              -- replace the feature columns with their natural log,
                          so the network is trained on ln(RV) instead of RV.
                          Mirrors the --log switch of HAR-RV_RUN.PY.
      drop_nonpositive -- drop rows whose target is <= 0. Implied by `log`,
                          since ln is undefined there. On EUR/USD RV these are
                          the two exchange holidays with no trading, which
                          HAR-RV_RUN.PY also drops; dropping them here keeps
                          the two models on exactly the same rows.

    drop_nonpositive must stay opt-in: targets like ETT oil temperature are
    legitimately negative, and dropping those rows would silently destroy the
    series.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, timeenc=0, freq='h',
                 log=False, drop_nonpositive=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.log = log
        # ln(x) needs x > 0, so --log always drops non-positive targets.
        self.drop_nonpositive = drop_nonpositive or log

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # guard against blank/incomplete trailing rows (e.g. Excel exports):
        # NaN targets would silently poison windows and test metrics
        df_raw = df_raw.dropna(subset=['date', self.target]).reset_index(drop=True)

        # Drop non-positive targets BEFORE the borders are computed, so the
        # split still counts real rows. Same rule as HAR-RV_RUN.PY.
        if self.drop_nonpositive:
            bad = df_raw[self.target] <= 0
            if bad.any():
                if self.flag == 'train':
                    print(f"  Dropped {int(bad.sum())} non-positive '"
                          f"{self.target}' row(s) (non-trading days): "
                          f"{list(df_raw.loc[bad, 'date'].astype(str))}")
                df_raw = df_raw[~bad].reset_index(drop=True)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        df_raw['date'] = pd.to_datetime(df_raw['date'])
        # train: 2012-2022, val: 2023-2024, test: 2025-end  (see year_split_borders)
        border1s, border2s = year_split_borders(
            df_raw['date'], self.seq_len, len(df_raw))
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # THE TRANSFORM. Applied to the feature columns themselves, so every
        # window the model ever sees -- input, label and target -- is ln(RV).
        # The horizon aggregation in Exp_Long_Term_Forecast._get_target then
        # runs on this scale.
        if self.log:
            nonpos = (df_data <= 0).any()
            if bool(nonpos.any()):
                raise ValueError(
                    f"--log needs strictly positive features, but column(s) "
                    f"{list(df_data.columns[nonpos])} contain values <= 0. "
                    f"Only the target column is filtered automatically; drop "
                    f"or fix the other columns, or run without --log.")
            df_data = np.log(df_data)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # Calendar dates of this split's rows, aligned 1:1 with data_x. Kept so
        # a forecast can be written out against the day it is FOR -- see
        # forecast_dates() -- which is what lets the deep models be joined to
        # HAR-RV_RUN.PY row for row.
        self.date_index = df_raw['date'][border1:border2].reset_index(drop=True)

    def forecast_dates(self):
        """
        The date each sample forecasts, one per __getitem__ index.

        Sample i reads rows [i, i+seq_len) and predicts rows
        [i+seq_len, i+seq_len+pred_len), so its information set ends the day
        BEFORE row i+seq_len and its target window opens ON row i+seq_len.
        Stamping the forecast with that opening day reproduces exactly the
        convention of HAR-RV_RUN.PY's build_horizon_target, where Y_t^(h)
        spans t .. t+h-1 off regressors shifted to t-1. The two models
        therefore key on identical dates with no offset to reconcile.

        Note this is independent of seq_len: a longer look-back moves the split
        border back by the same amount (year_split_borders), so the first
        forecast still lands on the first row of the split proper.
        """
        return self.date_index.iloc[self.seq_len:self.seq_len + len(self)] \
                   .reset_index(drop=True)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
