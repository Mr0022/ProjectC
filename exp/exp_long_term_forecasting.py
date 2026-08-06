from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, QLIKE, lognormal_back_transform
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
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
        # Standardisation constants, cached from the dataset by _get_data.
        # Stay None when the loader is not scaling (--scale 0), which turns
        # every _unscale below into a no-op.
        self._scale_mean = None
        self._scale_std = None

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

        When the loader standardises, the window arrives in z-space and is
        mapped back first -- neither the arithmetic mean of variances nor the
        log-of-mean is meaningful on standardised values.
        """
        y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        if getattr(self.args, 'aggregate_mean', False):
            y = self._unscale(y, f_dim)
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
        # Cache the standardisation constants on first use. Dataset_Custom fits
        # its scaler on the TRAIN split whatever `flag` asks for, so every split
        # carries the same constants and whichever loader is built first wins.
        stats = getattr(data_set, 'scaler_stats', None)
        if stats is not None and self._scale_mean is None:
            mean, std = stats
            self._scale_mean = torch.tensor(mean, dtype=torch.float32,
                                            device=self.device)
            self._scale_std = torch.tensor(std, dtype=torch.float32,
                                           device=self.device)
        return data_set, data_loader

    def _unscale(self, x, f_dim):
        """Undo the loader's standardisation, putting `x` back on the modelling
        scale -- raw RV, or ln(RV) under --log.

        Used only on the --aggregate_mean path. The HAR target
        ln(mean(exp(.))) lives on that scale and does NOT commute with an
        affine map, so the aggregation -- and every metric built on top of it,
        including the QLIKE floor and the lognormal back-transform -- has to
        happen after this. Without --aggregate_mean the loss stays in z-space
        exactly as upstream does it, and --inverse handles the test-time map.

        The transform is affine with a constant sigma, so the aggregated MSE is
        just sigma^2 times its z-space counterpart: gradients rescale
        uniformly and early stopping still selects the same epoch.
        """
        if self._scale_mean is None:
            return x
        return x * self._scale_std[f_dim:] + self._scale_mean[f_dim:]

    def _pred_to_target_scale(self, outputs, f_dim):
        """Put a forecast on the same scale as _get_target's return value.

        The models de-normalise their own per-window RevIN internally, so their
        output arrives in whatever space the loader handed them -- z-space when
        scaling is on. That is the loader's transform, not the model's, so it
        is ours to undo.
        """
        if getattr(self.args, 'aggregate_mean', False):
            return self._unscale(outputs, f_dim)
        return outputs

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
                outputs = self._pred_to_target_scale(outputs, f_dim)
                target = self._get_target(batch_y, f_dim)
                preds.append(outputs.detach().cpu().numpy().reshape(-1))
                trues.append(target.detach().cpu().numpy().reshape(-1))
        self.model.train()
        return np.concatenate(preds), np.concatenate(trues)

    def _report_rv_metrics(self, preds, trues, setting):
        """
        Report the test losses on the same footing as HAR-RV_RUN.PY, so this
        model and the HAR-RV baseline can be read side by side.

        raw mode -- preds/trues are already the h-day forward mean of RV, so
        MSE/MAE/QLIKE are raw-variance losses and line up with HAR-RV's
        raw-scale table directly. Forecasts arrive clipped at zero (see test),
        since a negative variance is not an admissible prediction; QLIKE is the
        exception, being infinite at zero, and floors non-positive forecasts at
        a small positive multiple of mean training RV instead. 'neg pred'
        counts those, so how much QLIKE leans on the floor stays visible.

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
        # data_x is standardised whenever the loader scales, and a floor is a
        # variance level, so undo that first -- on z-scores the mean collapses
        # to ~0 and the floor with it. Then keep only the target column, which
        # is the one the forecasts and QLIKE actually concern.
        train_data, _ = self._get_data(flag='train')
        train_series = np.asarray(train_data.data_x, dtype=float)
        if getattr(train_data, 'scaler_stats', None) is not None:
            train_series = train_data.inverse_transform(train_series)
        train_series = train_series[:, (-1 if self.args.features == 'MS' else 0):]
        mean_train_rv = float(np.mean(np.exp(train_series) if log_mode
                                      else train_series))
        floor = 1e-4 * mean_train_rv

        lines = []
        metrics = {}
        if not log_mode:
            q, n_bad = QLIKE(preds, trues, floor)
            metrics = {'MSE': float(np.mean((trues - preds) ** 2)),
                       'MAE': float(np.mean(np.abs(trues - preds))),
                       'QLIKE': q}
            lines += [f"  MSE        : {metrics['MSE']:.6f}",
                      f"  MAE        : {metrics['MAE']:.6f}",
                      f"  QLIKE      : {q:.6f}",
                      f"  neg pred   : {n_bad} ({100.0 * n_bad / len(preds):.1f}%)"]
        else:
            # Correction terms from TRAINING residuals only -- using test
            # residuals would leak the out-of-sample outcome into the forecast.
            #
            # Measured over a SEQUENTIAL pass of the whole training split, not
            # over train_loader: that one shuffles and drops its last partial
            # batch, so the moments would be estimated from a random subset
            # that changes with the RNG state, and two runs of the same command
            # would report different QLIKE / MSE_RV. The correction is part of
            # the forecast, so it has to be reproducible.
            full_train = DataLoader(train_data, batch_size=self.args.batch_size,
                                    shuffle=False, drop_last=False,
                                    num_workers=self.args.num_workers)
            tr_p, tr_t = self._forward_collect(full_train)
            resid = tr_t - tr_p
            bias = float(resid.mean())
            resid_var = float(resid.var(ddof=1))

            actual_rv = np.exp(trues)
            pred_rv = lognormal_back_transform(preds, resid_var, bias)
            pred_naive = lognormal_back_transform(preds)
            # QLIKE(pred, true): the ratio is actual/forecast, and the loss is
            # NOT symmetric in it -- feeding the inverse gives 1/r + ln r - 1,
            # a different loss that agrees with this one only to second order.
            # The raw branch above passes (forecast, actual) and so must this,
            # or a --log run cannot be set beside HAR-RV_RUN.PY, whose qlike()
            # takes (actual, predicted) and computes the same ratio.
            q, n_bad = QLIKE(pred_rv, actual_rv, floor)
            q_naive, _ = QLIKE(pred_naive, actual_rv, floor)
            shift = np.exp(bias + resid_var / 2.0)
            metrics = {'MSE [ln]': float(np.mean((trues - preds) ** 2)),
                       'MAE [ln]': float(np.mean(np.abs(trues - preds))),
                       'QLIKE [RV]': q,
                       'MSE_RV': float(np.mean((actual_rv - pred_rv) ** 2)),
                       'MAE_RV': float(np.mean(np.abs(actual_rv - pred_rv)))}
            lines += [
                f"  MSE  [ln]  : {metrics['MSE [ln]']:.6f}",
                f"  MAE  [ln]  : {metrics['MAE [ln]']:.6f}",
                f"  QLIKE [RV] : {q:.6f}   (naive exp: {q_naive:.6f})",
                f"  MSE_RV     : {metrics['MSE_RV']:.6f}",
                f"  MAE_RV     : {metrics['MAE_RV']:.6f}",
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
        return metrics

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
                outputs = self._pred_to_target_scale(outputs, f_dim)
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
                        outputs = self._pred_to_target_scale(outputs, f_dim)
                        batch_y = self._get_target(batch_y, f_dim)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    outputs = self._pred_to_target_scale(outputs, f_dim)
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
                    # Single-value forecast vs the log(mean(RV)) target. The
                    # generic --inverse path below cannot serve this branch: it
                    # un-scales a full pred_len window, while here the target is
                    # already aggregated. _pred_to_target_scale applies the same
                    # affine inverse to the scalar instead, so both sides reach
                    # the RV/ln(RV) scale the HAR metrics are defined on.
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    outputs = self._pred_to_target_scale(outputs, f_dim)
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

        # A variance forecast below zero is not admissible, so the model's best
        # feasible prediction is zero and the raw-scale errors are measured
        # against max(pred, 0). Only under --aggregate_mean WITHOUT --log are
        # the predictions variances; under --log they are ln(RV), where a
        # negative value is an ordinary variance below 1 and must be left
        # alone. Clipping here rather than inside the metric keeps the headline
        # mse:/mae: line, the HAR-comparable block and pred.npy consistent.
        # QLIKE is unaffected: it is infinite at zero, so it keeps its own
        # positive floor (see _report_rv_metrics).
        if getattr(self.args, 'aggregate_mean', False) and not getattr(self.args, 'log', False):
            n_clipped = int((preds < 0).sum())
            if n_clipped:
                print('clipped {} negative variance forecast(s) to 0 ({:.1f}%)'.format(
                    n_clipped, 100.0 * n_clipped / preds.size))
            preds = np.maximum(preds, 0.0)

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
        # Returned so run.py can average them over the --itr repeats. Under
        # --aggregate_mean the HAR-comparable set replaces the plain one rather
        # than extending it: it already carries MSE and MAE, on the scale the
        # block above reports them, and adds QLIKE.
        if getattr(self.args, 'aggregate_mean', False):
            run_metrics = self._report_rv_metrics(preds, trues, setting)
        else:
            run_metrics = {'MSE': float(mse), 'MAE': float(mae)}

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return run_metrics
