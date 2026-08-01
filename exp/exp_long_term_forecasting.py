from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, QLIKE, lognormal_back_transform
from utils.forecast_export import (build_forecast_frame, write_forecast_frame,
                                   run_tag)
import torch
import torch.nn as nn
from torch import optim
import os
import time
import math
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        # Mean-aggregation: the model forecasts a SINGLE value -- the horizon
        # average of RV (or its log). Build the head with target_window=1 by
        # temporarily setting pred_len=1, then restore the original pred_len so
        # the data loader still returns the full future window (needed to build
        # the ground-truth mean in _get_target).
        if getattr(self.args, 'aggregate_mean', False):
            orig_pred_len = self.args.pred_len
            self.args.pred_len = 1
            model = self.model_dict[self.args.model](self.args).float()
            self.args.pred_len = orig_pred_len
        else:
            model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_target(self, batch_y, f_dim):
        """Slice the future window and, when aggregating, reduce it to the
        HAR-RV target Y^(h) over the pred_len horizon.

        The aggregation ALWAYS happens in variance space -- the additive space,
        where averaging variances is the meaningful operation -- and the log,
        when there is one, is applied to the aggregate:

            raw mode :  Y^(h) =     (1/h) * Sum_k RV_(t+k)
            --log    :  Y^(h) = ln( (1/h) * Sum_k RV_(t+k) )

        These are exactly the two targets HAR-RV_RUN.PY builds, so a run with
        --aggregate_mean and pred_len = h is directly comparable to HAR-RV at
        horizon h on the matching scale.

        Under --log the series already holds ln(RV), so the log-of-mean is
        computed as
            ln(mean(exp(ln_RV))) == logsumexp(ln_RV) - ln(h)
        which is algebraically identical but avoids exponentiating a possibly
        large magnitude before summing. Note this is log-of-mean, NOT
        mean-of-logs: averaging the logs would target a geometric forward mean,
        a smaller and much smoother quantity that would not line up with the
        raw-mode target or with HAR-RV.
        """
        y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        if getattr(self.args, 'aggregate_mean', False):
            h = y.shape[1]
            if getattr(self.args, 'log', False):
                # series is ln(RV): ln(mean(RV)) = logsumexp(ln_RV) - ln(h)
                y = torch.logsumexp(y, dim=1, keepdim=True) - math.log(h)
            else:
                # series is raw RV: plain arithmetic mean over the horizon
                y = y.mean(dim=1, keepdim=True)
        return y

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _forward_collect(self, loader):
        """Run the model over `loader` and return (preds, trues) as flat arrays
        on the modelling scale. Used to measure the TRAINING residual variance
        that the --log Jensen correction needs."""
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                    dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target = self._get_target(batch_y, f_dim)
                preds.append(outputs.detach().cpu().numpy().reshape(-1))
                trues.append(target.detach().cpu().numpy().reshape(-1))
        self.model.train()
        return np.concatenate(preds), np.concatenate(trues)

    def _qlike_floor(self):
        """
        Floor for non-positive variance forecasts: 1e-4 * mean training RV,
        the rule HAR-RV_RUN.PY uses. Cached -- it rebuilds the training set.
        """
        if getattr(self, '_qlike_floor_cache', None) is None:
            train_data, _ = self._get_data(flag='train')
            train_series = np.asarray(train_data.data_x, dtype=float)
            mean_train_rv = float(np.mean(
                np.exp(train_series) if getattr(self.args, 'log', False)
                else train_series))
            self._qlike_floor_cache = 1e-4 * mean_train_rv
        return self._qlike_floor_cache

    def _log_correction(self):
        """
        (bias, resid_var) for the lognormal back-transform, measured on
        TRAINING residuals only -- test residuals would leak the out-of-sample
        outcome into the forecast. Returns (0.0, 0.0) in raw mode, where no
        back-transform happens. Cached: it costs a full pass over the training
        set, and both the metrics report and the forecast export need it, which
        also guarantees the two agree.
        """
        if not getattr(self.args, 'log', False):
            return 0.0, 0.0
        if getattr(self, '_log_correction_cache', None) is None:
            _, train_loader = self._get_data(flag='train')
            tr_p, tr_t = self._forward_collect(train_loader)
            resid = tr_t - tr_p
            self._log_correction_cache = (float(resid.mean()),
                                          float(resid.var(ddof=1)))
        return self._log_correction_cache

    def _export_forecasts(self, preds, trues, test_data, setting):
        """
        Write one row per test forecast, keyed by the date it is FOR.

        This is the artifact Diebold-Mariano and MCS actually consume: both
        need the per-observation loss series, which no aggregate in
        result_long_term_forecast.txt can reconstruct. Because the rows carry
        dates on HAR-RV_RUN.PY's convention, a deep model and the HAR baseline
        join directly -- see utils/forecast_export.py.

        Only meaningful under --aggregate_mean, where a run produces exactly
        one number per forecast date (the horizon average Y^(h)); the caller
        enforces that.
        """
        dates = test_data.forecast_dates()
        preds = np.asarray(preds, dtype=float).reshape(-1)
        trues = np.asarray(trues, dtype=float).reshape(-1)
        if len(dates) != len(preds):
            # Loud rather than silent: a mismatch here would misalign every
            # downstream test by an unknown offset.
            raise RuntimeError(
                f"{len(dates)} forecast dates but {len(preds)} forecasts for "
                f"{setting}. Test loader must run unshuffled with "
                f"drop_last=False for the dates to line up.")

        scale = 'log' if getattr(self.args, 'log', False) else 'raw'
        seed = getattr(self.args, 'fix_seed', 2021)
        bias, resid_var = self._log_correction()

        frame = build_forecast_frame(
            dates, preds, trues,
            model=self.args.model, scale=scale, horizon=self.args.pred_len,
            seed=seed, split='test', bias=bias, resid_var=resid_var)

        out_dir = getattr(self.args, 'forecast_dir', './forecasts')
        tag = getattr(self.args, 'run_tag', '') or run_tag(
            self.args.model, scale, self.args.pred_len, seed)
        path = write_forecast_frame(frame, out_dir, tag)
        print(f"  -> Saved {len(frame)} per-date forecasts: {path}")

    def _report_rv_metrics(self, preds, trues, setting):
        """
        Report the test losses on the same footing as HAR-RV_RUN.PY, so this
        model and the HAR-RV baseline can be read side by side.

        raw mode -- preds/trues are already the h-day forward mean of RV, so
        MSE/MAE/QLIKE are raw-variance losses and line up with HAR-RV's
        raw-scale table directly.

        --log -- preds/trues are ln(h-day forward mean of RV):
          * MSE/MAE are reported in ln(RV) units, the scale the network is
            trained on, and are NOT comparable to raw-scale losses.
          * QLIKE needs variances, so both sides are exponentiated first, the
            forecast with the lognormal Jensen correction exp(sigma^2/2) where
            sigma^2 is the TRAINING residual variance -- the same estimator
            HAR-RV_RUN.PY uses. QLIKE_naive omits the correction so its size
            stays visible.
          * MSE_RV/MAE_RV repeat the errors on that back-transformed variance
            scale. Because the target is ln(arithmetic forward mean), exp()
            recovers precisely the raw-mode target, so these ARE comparable to
            a raw run and to HAR-RV at every horizon.
        """
        log_mode = getattr(self.args, 'log', False)
        preds = np.asarray(preds, dtype=float).reshape(-1)
        trues = np.asarray(trues, dtype=float).reshape(-1)

        # QLIKE floor: same rule as HAR-RV_RUN.PY, 1e-4 * mean training RV.
        floor = self._qlike_floor()

        lines = []
        if not log_mode:
            q, n_bad = QLIKE(preds, trues, floor)
            lines += [f"  MSE        : {np.mean((trues - preds) ** 2):.6f}",
                      f"  MAE        : {np.mean(np.abs(trues - preds)):.6f}",
                      f"  QLIKE      : {q:.6f}",
                      f"  neg pred   : {n_bad} ({100.0 * n_bad / len(preds):.1f}%)"]
        else:
            # Correction terms from TRAINING residuals only -- using test
            # residuals would leak the out-of-sample outcome into the forecast.
            bias, resid_var = self._log_correction()

            actual_rv = np.exp(trues)
            pred_rv = lognormal_back_transform(preds, resid_var, bias)
            pred_naive = lognormal_back_transform(preds)
            q, n_bad = QLIKE(actual_rv, pred_rv, floor)
            q_naive, _ = QLIKE(actual_rv, pred_naive, floor)
            shift = np.exp(bias + resid_var / 2.0)
            lines += [
                f"  MSE  [ln]  : {np.mean((trues - preds) ** 2):.6f}",
                f"  MAE  [ln]  : {np.mean(np.abs(trues - preds)):.6f}",
                f"  QLIKE [RV] : {q:.6f}   (naive exp: {q_naive:.6f})",
                f"  MSE_RV     : {np.mean((actual_rv - pred_rv) ** 2):.6f}",
                f"  MAE_RV     : {np.mean(np.abs(actual_rv - pred_rv)):.6f}",
                f"  back-trans : bias={bias:+.6f}  sigma^2={resid_var:.6f}  -> "
                f"x{shift:.4f}",
                f"  neg pred   : {n_bad} (0 by construction under --log)"]

        scale = 'ln_RV' if log_mode else 'raw_RV'
        header = (f"  HAR-COMPARABLE TEST METRICS  [{scale}]  "
                  f"h = {self.args.pred_len}   n = {len(preds)}")
        block = "\n".join([header] + lines)
        print("\n" + "=" * 72 + "\n" + block + "\n" + "=" * 72)
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(block + "\n\n")

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = self._get_target(batch_y, f_dim)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = self._get_target(batch_y, f_dim)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = self._get_target(batch_y, f_dim)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                if getattr(self.args, 'aggregate_mean', False):
                    # single-value forecast vs log(mean(RV)) target; no inverse
                    # transform (aggregated target is not in the scaler's space)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = self._get_target(batch_y, f_dim)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                else:
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = batch_y.shape
                        if outputs.shape[-1] != batch_y.shape[-1]:
                            outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        # With --aggregate_mean the target is the HAR-RV target Y^(h), so also
        # report the losses the way HAR-RV_RUN.PY does (QLIKE, and the
        # back-transformed variance scale under --log) to make the two
        # directly comparable.
        if getattr(self.args, 'aggregate_mean', False):
            self._report_rv_metrics(preds, trues, setting)
            # Per-date forecasts for Diebold-Mariano / MCS. Written here rather
            # than derived later from pred.npy because only this scope knows
            # the dates and the training-residual correction.
            self._export_forecasts(preds, trues, test_data, setting)

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
