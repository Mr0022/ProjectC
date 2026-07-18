from data_provider.data_loader import Dataset_Custom, Dataset_Custom_Events, Dataset_Pred
from data_provider.data_loader import Dataset_HAR_Residual
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'custom_events': Dataset_Custom_Events,
    'har_residual': Dataset_HAR_Residual,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
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

    extra_kwargs = {}
    if Data is Dataset_Custom_Events:
        extra_kwargs['event_path'] = getattr(args, 'event_data_path', 'events.csv')
    elif Data is Dataset_Pred and getattr(args, 'use_events', False):
        raise NotImplementedError(
            'flag="pred" (Dataset_Pred) does not support --use_events; '
            'events beyond the dataset are unknown to Dataset_Pred')

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        **extra_kwargs
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
