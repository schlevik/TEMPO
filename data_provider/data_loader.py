import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from data_provider.m4 import M4Dataset, M4Meta
from utils.timefeatures import time_features
from utils.tools import convert_tsf_to_dataframe
import warnings
from pathlib import Path
import pickle
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')

stl_position = 'stl/'

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, data_name = 'etth2', max_len=-1, train_all=False):
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

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        
       
    def stl_resolve(self, data_raw, data_name):
        """
        STL Global Decomposition
        """
        # self.data_name = 'etth1'
        self.data_name = data_name
        save_stl = stl_position + self.data_name   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend.pk'
        seasonal_pk = self.save_stl + '/seasonal.pk'
        resid_pk = self.save_stl + '/resid.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            os.makedirs(self.save_stl, exist_ok=True)
            data_raw['date'] = pd.to_datetime(data_raw['date'])
            data_raw.set_index('date', inplace=True)

            [n,m] = data_raw.shape

            trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

            cols = data_raw.columns
            for i, col in enumerate(cols):
                df = data_raw[col]
                # df = df.resample(self.args.freq).mean().ffill()
                if 'weather' in self.data_name: # == 'weather':
                    res = STL(df, period = 24*6).fit()
                elif 'ill' in self.data_name: #== :
                    res = STL(df, period = 7).fit()
                elif 'etth1' in self.data_name or 'etth2' in self.data_name:
                    res = STL(df, period = 24).fit()
                else:
                    res = STL(df, period = 24*2).fit()

                trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
                resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

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

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        # After we get data, we do the stl resolve
        col_date = df_raw.columns[:1]
        df_time = df_raw[col_date]
        data_raw = pd.DataFrame.join(df_time, pd.DataFrame(data))#[border1:border2]
        trend_stamp, seasonal_stamp, resid_stamp = self.stl_resolve(data_raw=data_raw, data_name=self.data_name)
        # end -dove

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

        self.trend_stamp = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp = resid_stamp[border1:border2]
    
    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_trend = self.trend_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_resid = self.resid_stamp[s_begin:s_end, feat_id:feat_id+1]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 percent=100, max_len=-1, data_name = 'ettm2', train_all=False):
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
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def stl_resolve(self, data_raw, data_name):
        """
        STL Global Decomposition
        """
        # self.data_name = 'etth1'
        self.data_name = data_name
        save_stl = stl_position + self.data_name   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend.pk'
        seasonal_pk = self.save_stl + '/seasonal.pk'
        resid_pk = self.save_stl + '/resid.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            os.makedirs(self.save_stl, exist_ok=True)
            data_raw['date'] = pd.to_datetime(data_raw['date'])
            data_raw.set_index('date', inplace=True)

            [n,m] = data_raw.shape

            trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

            cols = data_raw.columns
            for i, col in enumerate(cols):
                df = data_raw[col]
                # df = df.resample(self.args.freq).mean().ffill()
                if 'weather' in self.data_name: # == 'weather':
                    res = STL(df, period = 24*6).fit()
                elif 'ill' in self.data_name:
                    res = STL(df).fit() # , period = 7 52？
                elif 'etth1' in self.data_name or 'etth2' in self.data_name:
                    res = STL(df, period = 24).fit()
                elif 'ettm1' in self.data_name or 'ettm2' in self.data_name:
                    res = STL(df, period = 24*4).fit()
                else:
                    res = STL(df).fit()
                
                
                trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
                resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

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

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
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

        # After we get data, we do the stl resolve
        col_date = df_raw.columns[:1]
        df_time = df_raw[col_date]
        data_raw = pd.DataFrame.join(df_time, pd.DataFrame(data))#[border1:border2]
        trend_stamp, seasonal_stamp, resid_stamp = self.stl_resolve(data_raw=data_raw, data_name=self.data_name)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.trend_stamp = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp = resid_stamp[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_trend = self.trend_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_resid = self.resid_stamp[s_begin:s_end, feat_id:feat_id+1]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 aug='national_illness_168_gen_50ksteps_10k.npy',
                 aug_only=False,
                 target='OT', scale=False, timeenc=0, freq='h',
                 percent=10, data_name = 'weather', max_len=-1, train_all=False):
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
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name
        self.aug_path = aug
        self.aug_only = aug_only
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        # self.save_stl = 'stl/' 
     

    def stl_resolve(self, data_raw):
        """
        STL Global Decomposition
        """
        
        save_stl = stl_position +  self.data_name   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend.pk'
        seasonal_pk = self.save_stl + '/seasonal.pk'
        resid_pk = self.save_stl + '/resid.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            os.makedirs(self.save_stl, exist_ok=True)
            data_raw['date'] = pd.to_datetime(data_raw['date'])
            data_raw.set_index('date', inplace=True)

            [n,m] = data_raw.shape

            trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

            cols = data_raw.columns
            for i, col in enumerate(cols):
                df = data_raw[col]
                # df = df.resample(self.args.freq).mean().ffill()
                print(self.data_name)
                # assert False
                
                if 'weather' in self.data_name: # == 'weather':
                    res = STL(df, period = 24*6).fit()
                elif 'ILI' in self.data_name:
                    res = STL(df).fit() # , period = 7 52？
                elif 'etth1' in self.data_name or 'etth2' in self.data_name:
                    res = STL(df, period = 24).fit()
                elif 'ettm1' in self.data_name or 'ettm2' in self.data_name:
                    res = STL(df, period = 24*4).fit()
                elif 'traffic' in self.data_name  or 'electricity' in self.data_name:
                    res = STL(df, period = 24).fit()
                else:
                    res = STL(df).fit()
                
                trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
                resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp

    def stl_resolve_aug(self, data_raw):
        """
        STL Global Decomposition
        """
        
        save_stl = stl_position +  self.data_name   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend_aug.pk'
        seasonal_pk = self.save_stl + '/seasonal_aug.pk'
        resid_pk = self.save_stl + '/resid_aug.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            trend_stamp = []
            seasonal_stamp = []
            resid_stamp = []
            print('decomposing time-series...')
            for i, df in tqdm(enumerate(data_raw)):
                # print(len(df))
                res = STL(df, period = 52*2).fit()
                
                trend_stamp.append(res.trend)
                seasonal_stamp.append(res.seasonal)
                resid_stamp.append(res.resid)
            print('done!')
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return torch.Tensor(np.stack(trend_stamp)), torch.Tensor(np.stack(seasonal_stamp)), torch.Tensor(np.stack(resid_stamp))

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

       

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        
       

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

        # After we get data, we do the stl resolve
        col_date = df_raw.columns[:1]
        df_time = df_raw[col_date]
        data_raw = pd.DataFrame.join(df_time, pd.DataFrame(data))#[border1:border2]
        trend_stamp, seasonal_stamp, resid_stamp = self.stl_resolve(data_raw=data_raw)
        
        

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
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
        self.trend_stamp = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp = resid_stamp[border1:border2]
        self.data_stamp = data_stamp

        # data aug stuff
        if self.aug_path and self.set_type == 0:
            self.aug = np.load(self.aug_path).squeeze()
        else:
            self.aug = None
        self.ds_len = (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.data_x.shape[-1]
        self.aug_num = len(self.aug) if self.aug is not None else 0
        self.aug_len = self.aug.shape[1] if self.aug is not None else 0
        print(self.aug_len, self.seq_len + self.label_len + self.pred_len)
        if self.aug_len:
            assert self.seq_len + self.label_len + self.pred_len <= self.aug_len, (self.seq_len + self.label_len + self.pred_len, self.aug_len)
        if self.set_type == 0:
            assert self.aug_len
        if self.aug_len:
            self.trend_stamp_aug, self.seasonal_stamp_aug, self.resid_stamp_aug = self.stl_resolve_aug(data_raw=self.aug)

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        # print(index, s_begin, self.ds_len, len(self))
        if index < self.ds_len and (not self.aug_only or self.set_type > 0):        
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_trend = self.trend_stamp[s_begin:s_end, feat_id:feat_id+1]
            seq_seasonal = self.seasonal_stamp[s_begin:s_end, feat_id:feat_id+1]
            seq_resid = self.resid_stamp[s_begin:s_end, feat_id:feat_id+1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            is_aug=False
        else:
            if not self.aug_only:
                index -= self.ds_len
            total_len = self.seq_len + self.label_len + self.pred_len
            # print(self.aug, self.set_type)
            sampled_timeseries = self.aug[index]
            sampled_trend = self.trend_stamp_aug[index]
            sampled_seasonal = self.seasonal_stamp_aug[index]
            sampled_resid = self.resid_stamp_aug[index]
            s_begin = np.random.randint(low=0,
                                      high=len(sampled_timeseries) - total_len,
                                      size=1)[0]

            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = sampled_timeseries[s_begin:s_end].reshape(-1, 1)
            seq_y = sampled_timeseries[r_begin:r_end].reshape(-1, 1)
            
            seq_trend = sampled_trend[s_begin:s_end].reshape(-1, 1)
            seq_seasonal = sampled_seasonal[s_begin:s_end].reshape(-1, 1)
            seq_resid = sampled_resid[s_begin:s_end].reshape(-1, 1)
            
            # seq_x_mark = self.data_stamp[s_begin:s_end]
            # seq_y_mark = self.data_stamp[r_begin:r_end]
            seq_x_mark = np.zeros((seq_x.shape[0], 4)).squeeze()
            seq_y_mark = np.zeros((seq_y.shape[0], 4)).squeeze()
            is_aug=True
        # if self.set_type == 0:
        #     print(seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape, is_aug, index, seq_trend.shape, seq_seasonal.shape, seq_resid.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid

    def __len__(self):
        # return 1000 #(
        # print(len(self.data_x), self.seq_len, self.pred_len+1, self.enc_in)
        # return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
        if self.set_type > 0:
            return self.ds_len
        else:
            return (self.ds_len if not self.aug_only else 0) + self.aug_num

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None,
                 percent=None, train_all=False, period = 24, max_len=-1, data_name = 'weather'):
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
        self.period = period
        self.data_name  = data_name  
        self.__read_data__()

    def stl_resolve(self, data_raw, period = 24):
        """
        STL Global Decomposition
        """
        
        save_stl = stl_position +  self.data_name   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend.pk'
        seasonal_pk = self.save_stl + '/seasonal.pk'
        resid_pk = self.save_stl + '/resid.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            os.makedirs(self.save_stl, exist_ok=True)
            data_raw['date'] = pd.to_datetime(data_raw['date'])
            data_raw.set_index('date', inplace=True)

            [n,m] = data_raw.shape

            trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

            cols = data_raw.columns
            for i, col in enumerate(cols):
                df = data_raw[col]
                # df = df.resample(self.args.freq).mean().ffill()
                
                
                if 'weather' in self.data_name: # == 'weather':
                    res = STL(df, period = 24*6).fit()
                elif 'ill' in self.data_name:
                    res = STL(df).fit() # 
                elif 'etth1' in self.data_name or 'etth2' in self.data_name:
                    res = STL(df, period = 24).fit()
                elif 'ettm1' in self.data_name or 'ettm2' in self.data_name:
                    res = STL(df, period = 24*4).fit()
                elif 'traffic' in self.data_name  or 'electricity' in self.data_name:
                    res = STL(df, period = 24).fit()
                else:
                    res = STL(df, period = period).fit()
                
                trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
                resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp


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
        border1 = 0 #len(df_raw) - self.seq_len
        border2 = int(0.1*len(df_raw)) - self.seq_len + 1

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
        
        col_date = df_raw.columns[:1]
        df_time = df_raw[col_date]
        data_raw = pd.DataFrame.join(df_time, pd.DataFrame(data))#[border1:border2]
        trend_stamp, seasonal_stamp, resid_stamp = self.stl_resolve(data_raw=data_raw, period = self.period)
        
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
        self.trend_stamp = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp = resid_stamp[border1:border2]
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
        seq_trend = self.trend_stamp[s_begin:s_end]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end]
        seq_resid = self.resid_stamp[s_begin:s_end]
        return seq_x.reshape(-1, 1), seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, seq_trend.reshape(-1, 1), seq_seasonal.reshape(-1, 1), seq_resid.reshape(-1, 1)

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=True, timeenc=0, freq='Daily',
                 percent=10, max_len=-1, train_all=False):
        
        self.train_all = train_all
        
        self.seq_len = size[0]
        self.pred_len = size[2]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.percent = percent
        self.max_len = max_len
        if self.max_len == -1:
            self.max_len = 1e8

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()


    def __read_data__(self):
        df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(os.path.join(self.root_path,
                                                                                                                              self.data_path))
        self.freq = frequency
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        
        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len-self.pred_len
            if self.train_all:
                border1s = [0,          0,          train_len-self.seq_len]
                border2s = [train_len,  train_len,  _len]
            else:
                border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
                border2s = [train_len - self.pred_len,  train_len,                                _len]
            border2s[0] = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len
            # print("_len = {}".format(_len))
            
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            curr_len = max(0, curr_len)
            
            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len
            
        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        if self.train_all:
            border1s = [0,          0,          train_len-self.seq_len]
            border2s = [train_len,  train_len,  _len]
        else:
            border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
            border2s = [train_len - self.pred_len,  train_len,                                _len]
        border2s[0] = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        if self.set_type == 2:
            s_end = -self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        # if self.set_type == 2:
        #     print("data_x.shape = {}, data_y.shape = {}".format(data_x.shape, data_y.shape))

        return data_x, data_y, data_x, data_y

    def __len__(self):
        if self.set_type == 0:
            # return self.tot_len
            return min(self.max_len, self.tot_len)
        else:
            return self.tot_len


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 aug=None,
                 aug_only=False,
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly',
                 percent_aug=100):
        self.aug_path = aug.replace('Weekly', seasonal_patterns)
        self.aug_only = aug_only
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.percent_aug = percent_aug

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()
        
    def stl_resolve(self, data_raw, data_name):
            """
            STL Global Decomposition
            """
            # self.data_name = 'etth1'
            self.data_name = data_name
            save_stl = stl_position + self.data_name + self.seasonal_patterns
            # save_stl = 'stl/' + 'weather'   

            self.save_stl = save_stl
            trend_pk = self.save_stl + '/trend.pk'
            seasonal_pk = self.save_stl + '/seasonal.pk'
            resid_pk = self.save_stl + '/resid.pk'
            if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
                with open(trend_pk, 'rb') as f:
                    trend_stamp = pickle.load(f)
                with open(seasonal_pk, 'rb') as f:
                    seasonal_stamp = pickle.load(f)
                with open(resid_pk, 'rb') as f:
                    resid_stamp = pickle.load(f)
            else:
                os.makedirs(self.save_stl, exist_ok=True)
                # data_raw['date'] = pd.to_datetime(data_raw['date'])
                # data_raw.set_index('date', inplace=True)

                # [n,m] = data_raw.shape

                trend_stamp = []
                seasonal_stamp = []
                resid_stamp = []
                if self.seasonal_patterns == 'Yearly':
                    period = 10
                elif self.seasonal_patterns == 'Quarterly':
                    period = 4
                elif self.seasonal_patterns == 'Hourly':
                    period = 24
                elif self.seasonal_patterns == 'Weekly':
                    period = 52
                elif self.seasonal_patterns == 'Daily':
                    period = 7
                elif self.seasonal_patterns == 'Monthly':
                    period = 12
                # cols = data_raw.columns
                print('decomposing original time-series...')
                for i, df in tqdm(enumerate(data_raw)):
                    # print(len(df))
                    res = STL(df, period = period).fit()

                    
                    trend_stamp.append(res.trend)
                    seasonal_stamp.append(res.seasonal)
                    resid_stamp.append(res.resid)
                print('done!')

                with open(trend_pk, 'wb') as f:
                    pickle.dump(trend_stamp, f)
                with open(seasonal_pk, 'wb') as f:
                    pickle.dump(seasonal_stamp, f)
                with open(resid_pk, 'wb') as f:
                    pickle.dump(resid_stamp, f)
            return trend_stamp, seasonal_stamp, resid_stamp
    def stl_resolve_aug(self, data_raw):
        """
        STL Global Decomposition
        """
        
        save_stl = stl_position + self.data_name + self.seasonal_patterns   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl +  '/trend_aug.pk'
        seasonal_pk = self.save_stl + '/seasonal_aug.pk'
        resid_pk = self.save_stl + '/resid_aug.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            trend_stamp = []
            seasonal_stamp = []
            resid_stamp = []
            print('decomposing aug time-series...')
            if self.seasonal_patterns == 'Yearly':
                period = 10
            elif self.seasonal_patterns == 'Quarterly':
                period = 4
            elif self.seasonal_patterns == 'Hourly':
                period = 24
            elif self.seasonal_patterns == 'Weekly':
                period = 52
            elif self.seasonal_patterns == 'Daily':
                period = 7
            elif self.seasonal_patterns == 'Monthly':
                period = 12
            for i, df in tqdm(enumerate(data_raw)):
                # print(len(df))
                res = STL(df, period = period).fit()
                
                trend_stamp.append(res.trend)
                seasonal_stamp.append(res.seasonal)
                resid_stamp.append(res.resid)
            print('done!')
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp
    #torch.Tensor(np.stack(trend_stamp)), torch.Tensor(np.stack(seasonal_stamp)), torch.Tensor(np.stack(resid_stamp))
    
    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)

       

        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]
        self.trend_stamp, self.seasonal_stamp, self.resid_stamp = self.stl_resolve(self.timeseries, 'm4')
        
        # data aug stuff
        if self.aug_path and self.flag == 'train':
            self.aug = np.load(self.aug_path).squeeze()
            print(self.percent_aug, len(self.timeseries))
            if self.percent_aug > 0:
                num_aug = int(self.percent_aug /100 * len(self.aug))
            else:
                num_aug = int(- self.percent_aug /100 * len(self.timeseries))
            print(num_aug)
            if num_aug < len(self.aug):
                self.aug = self.aug[np.random.choice(len(self.aug), num_aug, replace=False)]
            print(f"New dataset size: {len(self.aug)}")
        else:
            self.aug = None
        self.trend_stamp_aug, self.seasonal_stamp_aug, self.resid_stamp_aug = self.stl_resolve_aug(self.aug)

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        trendsample = np.zeros((self.seq_len, 1))
        seasonsample = np.zeros((self.seq_len, 1))
        residsample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset
        if index < len(self.timeseries):
            sampled_timeseries = self.timeseries[index]
            trend = self.trend_stamp[index]
            season = self.seasonal_stamp[index]
            resid = self.resid_stamp[index]
        else:
            index = index - len(self.timeseries)
            sampled_timeseries = self.aug[index]
            trend = self.trend_stamp_aug[index]
            season = self.seasonal_stamp_aug[index]
            resid = self.resid_stamp_aug[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        
        trendsample_window = trend[max(0, cut_point - self.seq_len):cut_point]
        trendsample[-len(trendsample_window):, 0] = trendsample_window
        
        seasonsample_window = season[max(0, cut_point - self.seq_len):cut_point]
        seasonsample[-len(seasonsample_window):, 0] = seasonsample_window
       
        residsample_window = resid[max(0, cut_point - self.seq_len):cut_point]
        residsample[-len(residsample_window):, 0] = residsample_window
       
        
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        

        return insample, outsample, insample_mask, outsample_mask, trendsample, seasonsample, residsample

    def __len__(self):
        return len(self.timeseries) + (len(self.aug) if self.aug is not None else 0)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        trendsample = np.zeros((len(self.timeseries), self.seq_len))
        seasonsample = np.zeros((len(self.timeseries), self.seq_len))
        residsample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            
            trendsample_last_window = self.trend_stamp[i][-self.seq_len:]
            trendsample[i, -len(ts):] = trendsample_last_window
            
            seasonsample_last_window = self.seasonal_stamp[i][-self.seq_len:]
            seasonsample[i, -len(ts):] = seasonsample_last_window
            
            residsample_last_window = self.resid_stamp[i][-self.seq_len:]
            residsample[i, -len(ts):] = residsample_last_window
            
            insample_mask[i, -len(ts):] = 1.0
            # todo
        return insample, insample_mask, trendsample, seasonsample, residsample

