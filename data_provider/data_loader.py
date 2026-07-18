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





class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, timeenc=0, freq='h'):
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

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

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

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        df_raw['date'] = pd.to_datetime(df_raw['date'])
        # train: 2010-2021, val: 2022-2023, test: 2024-2025
        train_end = int((df_raw['date'].dt.year <= 2021).sum())
        val_end = int((df_raw['date'].dt.year <= 2023).sum())
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
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
    

class Dataset_Custom_Events(Dataset_Custom):
    """
    Dataset_Custom + a daily macro news-event calendar (data/events.csv).

    The event file must contain a 'date' column plus numeric per-day event
    features (multi-hot 'evt_*' indicator columns and 'n_events*' counts).
    Rows are aligned to the target CSV's trading dates by date; days missing
    from the event file are treated as no-event days (all zeros).

    'evt_*' indicator columns are kept raw (0/1). All other event columns
    (counts) are standardised with statistics from the TRAIN years only,
    mirroring how the target series is scaled.

    __getitem__ additionally returns:
        seq_x_events : (seq_len,  n_event_features)  events on the look-back days
        seq_y_events : (pred_len, n_event_features)  the KNOWN event schedule for
                                                     the pred_len forecast days
    Both are what is known at the forecast origin: the future window encodes
    only that an event is scheduled (release calendar), never its outcome.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='realized_volatility.csv',
                 target='ln_RV', scale=False, timeenc=0, freq='h',
                 event_path='events.csv'):
        self.event_path = event_path
        super().__init__(root_path=root_path, flag=flag, size=size,
                         features=features, data_path=data_path,
                         target=target, scale=scale, timeenc=timeenc, freq=freq)

    def __read_data__(self):
        super().__read_data__()

        # re-read and filter exactly like Dataset_Custom so row indices align
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = df_raw.dropna(subset=['date', self.target]).reset_index(drop=True)
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        df_ev = pd.read_csv(os.path.join(self.root_path, self.event_path))
        df_ev['date'] = pd.to_datetime(df_ev['date'])
        ev_cols = [c for c in df_ev.columns if c != 'date']

        # align event rows to the trading dates of the target series;
        # dates absent from the event file become all-zero (no-event) days
        df_ev = df_ev.drop_duplicates(subset='date').set_index('date')
        events = df_ev.reindex(df_raw['date']).fillna(0.0)[ev_cols].values.astype(np.float32)

        # standardise count columns on the TRAIN years only; keep evt_* binary
        train_end = int((df_raw['date'].dt.year <= 2021).sum())
        count_idx = [i for i, c in enumerate(ev_cols) if not c.startswith('evt_')]
        if count_idx:
            tr = events[:train_end, count_idx]
            mean, std = tr.mean(axis=0), tr.std(axis=0) + 1e-8
            events[:, count_idx] = (events[:, count_idx] - mean) / std

        self.event_cols = ev_cols
        self.n_event_features = events.shape[1]
        # slice with the same borders as data_x/data_y so indices line up
        val_end = int((df_raw['date'].dt.year <= 2023).sum())
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.data_events = events[border1:border2]

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = super().__getitem__(index)

        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x_events = self.data_events[s_begin:s_end]
        seq_y_events = self.data_events[s_end:s_end + self.pred_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_events, seq_y_events


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
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
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


class Dataset_HAR_Residual(Dataset):
    """
    HAR-LSTM residual dataset (Corsi 2009 + LSTM correction).

    The HAR-RV linear model is fit by OLS on the TRAINING window only
    (2010-2021, mirroring Dataset_Custom). Its prediction of the h-day
    forward average log-RV is removed, and the LSTM is trained to forecast
    the HAR *residual* from a look-back window of observed ln(RV). The final
    hybrid forecast, reconstructed at test time, is:

        y_hat^(h) = HAR_pred^(h) + LSTM_residual_pred

    Design notes
    ------------
    * Horizon h is taken from pred_len. The target Y_t^(h) is the h-day
      forward mean of ln(RV) -- identical to the aggregate-mean target used
      by the deep-learning baselines, so the comparison is apples-to-apples.
    * The LSTM input window is the *observed* ln(RV) series (known at the
      forecast origin t), so there is no look-ahead from overlapping
      residuals.
    * Inputs and residual targets are standardised with statistics computed
      on the TRAIN split only; targets are de-standardised before the HAR
      prediction is added back. RevIN is therefore unnecessary (use --revin 0).

    HAR regressors (information available AT the forecast origin t, i.e.
    including today's RV_t -- the same information set as the deep look-back
    window, which also ends at y[t]):
        RV_d = ln(RV_t)
        RV_w = mean(ln(RV_t) .. ln(RV_{t-4}))
        RV_m = mean(ln(RV_t) .. ln(RV_{t-21}))

    __getitem__ returns four (1-channel) tensors:
        seq_x        : (seq_len, 1)  standardised look-back window of ln(RV)
        resid_target : (1, 1)        standardised HAR residual at origin t
        har_pred     : (1, 1)        HAR linear prediction (original scale)
        y_true       : (1, 1)        actual Y_t^(h) (original scale)
    """

    LAG_W = 5    # weekly component window
    LAG_M = 22   # monthly component window

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='realized_volatility.csv',
                 target='ln_RV', scale=True, timeenc=0, freq='d'):
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 48, 1
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        h = self.pred_len  # forecast horizon (steps ahead)

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        date_col = 'date' if 'date' in df_raw.columns else df_raw.columns[0]
        df_raw[date_col] = pd.to_datetime(df_raw[date_col])
        df_raw = df_raw.sort_values(date_col).reset_index(drop=True)

        if self.target in df_raw.columns:
            val_col = self.target
        elif 'ln_RV' in df_raw.columns:
            val_col = 'ln_RV'
        else:
            val_col = df_raw.select_dtypes('number').columns[0]
        y = df_raw[val_col].astype(float).values
        N = len(y)

        # ── HAR regressors (info AT origin t => include today's RV_t) ────────
        #    This matches the deep look-back window, which also ends at y[t],
        #    so the HAR linear part and the residual learner share one
        #    information set (Corsi 2009 convention; no look-ahead).
        s = pd.Series(y)
        RV_d = s.values
        RV_w = s.rolling(self.LAG_W).mean().values
        RV_m = s.rolling(self.LAG_M).mean().values

        # ── Horizon target Y_t^(h) = mean(ln_RV[t+1 .. t+h]) ─────────────────
        Yh = s.rolling(h).mean().shift(-h).values

        # ── Split borders (mirror Dataset_Custom; val kept separate) ─────────
        years = df_raw[date_col].dt.year.values
        train_end = int((years <= 2021).sum())
        val_end = int((years <= 2023).sum())
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, N]

        feat_valid = (~np.isnan(RV_d)) & (~np.isnan(RV_w)) & (~np.isnan(RV_m))

        # ── Fit HAR-RV by OLS on the TRAIN window only (2010-2021) ───────────
        train_mask = np.zeros(N, dtype=bool)
        train_mask[:train_end] = True
        fit_mask = train_mask & feat_valid & (~np.isnan(Yh))
        X_all = np.column_stack([np.ones(N), RV_d, RV_w, RV_m])
        beta, *_ = np.linalg.lstsq(X_all[fit_mask], Yh[fit_mask], rcond=None)
        self.har_beta = beta  # [const, b_d, b_w, b_m]

        har_pred = X_all @ beta
        har_pred[~feat_valid] = np.nan
        resid = Yh - har_pred

        # ── Standardisation statistics from TRAIN split only ─────────────────
        if self.scale:
            x_tr = y[:train_end]
            self.x_mean, self.x_std = float(np.mean(x_tr)), float(np.std(x_tr) + 1e-8)
            r_tr = resid[fit_mask]
            self.r_mean, self.r_std = float(np.mean(r_tr)), float(np.std(r_tr) + 1e-8)
        else:
            self.x_mean, self.x_std = 0.0, 1.0
            self.r_mean, self.r_std = 0.0, 1.0

        # ── Materialise windowed samples for the requested split ─────────────
        b1, b2 = border1s[self.set_type], border2s[self.set_type]
        seqx, rtar, hpred, ytrue, idxs = [], [], [], [], []
        first_origin = b1 + self.seq_len - 1
        for g in range(first_origin, b2):
            if g - self.seq_len + 1 < 0:
                continue
            if np.isnan(resid[g]) or np.isnan(har_pred[g]) or np.isnan(Yh[g]):
                continue
            window = (y[g - self.seq_len + 1: g + 1] - self.x_mean) / self.x_std
            seqx.append(window.reshape(self.seq_len, 1))
            rtar.append([[(resid[g] - self.r_mean) / self.r_std]])
            hpred.append([[har_pred[g]]])
            ytrue.append([[Yh[g]]])
            idxs.append(g)

        self.seq_x = np.asarray(seqx, dtype=np.float32)
        self.resid = np.asarray(rtar, dtype=np.float32)
        self.har = np.asarray(hpred, dtype=np.float32)
        self.ytrue = np.asarray(ytrue, dtype=np.float32)
        self.origin_idx = np.asarray(idxs)
        self.dates = df_raw[date_col].values

    def __getitem__(self, i):
        return self.seq_x[i], self.resid[i], self.har[i], self.ytrue[i]

    def __len__(self):
        return len(self.seq_x)

    def inverse_resid(self, r_std_pred):
        """De-standardise a residual prediction back to ln(RV) scale."""
        return r_std_pred * self.r_std + self.r_mean
