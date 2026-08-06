from data_provider.data_loader import (Dataset_Custom, Dataset_Forex,
                                       Dataset_Crypto, Dataset_Pred)
from torch.utils.data import DataLoader

# The two asset classes differ only in their train/val/test calendar; see the
# FOREX and CRYPTO sections of data_loader.py.
#   forex   train 2012-01..2022-12 | val 2023-01..2024-12 | test 2025-01..end
#   crypto  train 2018-06..2024-06 | val 2024-07..2025-06 | test 2025-07..2026-06
# 'custom' is kept as an alias of 'forex' so existing FX scripts run unchanged.
data_dict = {
    'custom': Dataset_Forex,
    'forex': Dataset_Forex,
    'crypto': Dataset_Crypto,
}


def data_provider(args, flag):
    if args.data not in data_dict:
        raise KeyError(
            f"unknown --data '{args.data}'; choose one of "
            f"{', '.join(sorted(data_dict))}")
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        # Keep the trailing partial batch. Dropping it silently discards up to
        # batch_size-1 test targets, which both understates the test set and
        # stops the metrics lining up row-for-row with HAR-RV_RUN.PY.
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        # Train-split StandardScaler, on by default like the Time-Series-Library.
        scale=bool(getattr(args, 'scale', 1)),
    )
    # Realized-variance options; only the Dataset_Custom family understands
    # them, and both default to off so every other dataset behaves as before.
    if issubclass(Data, Dataset_Custom):
        kwargs['log'] = getattr(args, 'log', False)
        kwargs['drop_nonpositive'] = getattr(args, 'drop_nonpositive', False)

    data_set = Data(**kwargs)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
