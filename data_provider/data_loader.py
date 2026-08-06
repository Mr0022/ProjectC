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
# Chronological train / val / test split.
#
# The two asset classes in data/ do not share a calendar: the FX series start in
# 2012 and split on year boundaries, the crypto series start mid-2018 and split
# on mid-year boundaries. Each gets its own dataset class below -- SECTION 1
# (Dataset_Forex) and SECTION 2 (Dataset_Crypto) -- differing only in which
# SplitSpec it carries:
#
#   forex   train 2012-01 .. 2022-12 | val 2023-01 .. 2024-12 | test 2025-01 .. end
#   crypto  train 2018-06 .. 2024-06 | val 2024-07 .. 2025-06 | test 2025-07 .. 2026-06
#
# The boundaries themselves live in data_provider/splits.py, which HAR-RV_RUN.PY
# imports as well -- edit them there and both model families follow.
# ==============================================================================
from data_provider.splits import (SplitSpec, month_split_borders,   # noqa: F401
                                  FOREX_SPLIT, CRYPTO_SPLIT,
                                  TRAIN_END_YEAR, VAL_END_YEAR)


class Dataset_Custom(Dataset):
    """
    Generic CSV dataset with a chronological calendar-based split.

    This is the shared base class. It holds everything that is the same for
    every asset class -- CSV reading, the log / drop_nonpositive options, the
    train-fit StandardScaler, the time stamps and the sliding windows -- and
    takes the split boundaries from the class attribute `SPLIT`. The asset
    classes in SECTION 1 (Dataset_Forex) and SECTION 2 (Dataset_Crypto) below
    are thin subclasses that swap that one attribute; instantiate one of those,
    not this class.

    scale -- standardise the feature columns with a StandardScaler fit on the
             TRAIN split only, exactly as the Time-Series-Library does, and ON
             by default to match upstream. When `log` is also set the order is
             ln first, then standardise, so the scaler sees ln(RV).

             The transform is applied to the SERIES, so everything this class
             emits -- data_x and data_y -- is in z-space. Consumers that need
             the modelling scale back (the HAR-comparable aggregation and
             metrics in Exp_Long_Term_Forecast) must undo it; see
             `scaler_stats` and `inverse_transform`.

    Two further options exist for realized-variance work, both off by default
    so that every other dataset (ETT, Weather, Traffic, ...) is untouched:

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

    # Split boundaries for this asset class; set by the subclasses below.
    SPLIT = None

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
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

    def _report_split(self, dates, border1s, border2s):
        """Print the calendar window this split actually covers, and fail loudly
        if it is too short to yield a single (seq_len + pred_len) window.

        A split that is empty or short is almost always a data/boundary mismatch
        -- the wrong asset class for the file, or a CSV that stops before the
        test window opens. Left unchecked it produces a silently empty loader
        and a confusing error much further downstream.
        """
        b1, b2 = border1s[self.set_type], border2s[self.set_type]
        n = b2 - b1
        # Rows the val/test windows borrow from the previous split purely as
        # look-back context; they are never scored as targets.
        context = 0 if self.set_type == 0 else border2s[self.set_type - 1] - b1
        need = self.seq_len + self.pred_len

        if n < need:
            raise ValueError(
                f"'{self.flag}' split of {self.data_path} has {n} row(s) on the "
                f"'{self.SPLIT.name}' calendar, but seq_len + pred_len = "
                f"{self.seq_len} + {self.pred_len} = {need} are needed for one "
                f"window. Check that the file covers the split window and that "
                f"--data matches the asset class of the file.")

        span = pd.to_datetime(pd.Series(np.asarray(dates)))
        print(f"  [{self.SPLIT.name}] {self.flag:<5} "
              f"{span.iloc[b1].date()} -> {span.iloc[b2 - 1].date()}  "
              f"({n} rows, {context} look-back + {n - context} scored, "
              f"{n - need + 1} windows)")

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
        # Boundaries come from this class's SPLIT -- see the FOREX and CRYPTO
        # sections further down this file.
        if self.SPLIT is None:
            raise TypeError(
                f"{type(self).__name__} has no SPLIT calendar. Use "
                f"Dataset_Forex ('--data forex' / '--data custom') or "
                f"Dataset_Crypto ('--data crypto'), or subclass this class "
                f"with a SplitSpec of your own.")
        border1s, border2s = month_split_borders(
            df_raw['date'], self.seq_len, len(df_raw), self.SPLIT)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self._report_split(df_raw['date'], border1s, border2s)

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

    @property
    def scaler_stats(self):
        """(mean, std) of the fitted scaler, or None when `scale` is off.

        The scaler is always fit on the TRAIN split whatever `flag` this
        instance was built with, so the train, val and test instances carry
        identical constants and either one may be used to undo the transform.
        """
        if not self.scale:
            return None
        return self.scaler.mean_, self.scaler.scale_


# ==============================================================================
# SECTION 1 -- FOREX     data/EURUSD-RV.csv, AUDUSD, EURGBP, USDCHF, USDJPY
#
#     train : start of sample  ..  2022-12
#     val   : 2023-01          ..  2024-12
#     test  : 2025-01          ..  end of sample
#
# Both ends stay open: training starts at whatever the file starts at, and the
# test window runs to the last row, so appending FX data extends the test set.
# Boundaries: FOREX_SPLIT in data_provider/splits.py.
# ==============================================================================

class Dataset_Forex(Dataset_Custom):
    """FX realized-variance CSVs. Year-boundary split, both ends open.

    This is what '--data custom' has always done, so 'custom' stays registered
    as an alias of this class and existing FX scripts run unchanged.
    """

    SPLIT = FOREX_SPLIT


# ==============================================================================
# SECTION 2 -- CRYPTO    data/btcusdt-RV.csv, ethusdt, adausdt, bnbusdt, xrpusdt
#
# Crypto starts mid-2018 and trades every calendar day, so the windows are set
# at mid-year boundaries instead:
#
#     train : 2018-06  ..  2024-06
#     val   : 2024-07  ..  2025-06
#     test  : 2025-07  ..  2026-06
#
# Unlike forex, BOTH ends are closed. Rows before 2018-06 or after 2026-06 are
# excluded from every split rather than folded into train or test, so extending
# a crypto CSV does not silently change what the test metrics cover.
# Boundaries: CRYPTO_SPLIT in data_provider/splits.py.
# ==============================================================================

class Dataset_Crypto(Dataset_Custom):
    """Crypto realized-variance CSVs. Mid-year split, both ends closed."""

    SPLIT = CRYPTO_SPLIT


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
