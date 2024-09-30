from data_provider.data_loader import Dataset_Custom, Dataset_Pred, Dataset_TSF, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_M4
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'm4': Dataset_M4,
}


def data_provider(args, flag, drop_last_test=True, train_all=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    max_len = args.max_len

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            aug_only=args.aug_only,
            aug=args.aug,
            percent_aug=args.percent_aug
        )
    elif args.aug:
        print("Data aug...")
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            max_len=max_len,
            train_all=train_all,
            data_name = args.data_name,
            percent_aug=args.percent_aug,
            aug=args.aug,
            aug_only=args.aug_only
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            max_len=max_len,
            train_all=train_all,
            data_name = args.data_name
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=min(batch_size, len(data_set)),
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    # if flag=='test':
        # print('test', len(data_set), len(data_loader), batch_size, min(batch_size, len(data_set)))
        # assert False
    return data_set, data_loader
