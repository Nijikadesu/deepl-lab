import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ------------------------------
# 1. 数据读取与预处理（不采样）
# ------------------------------
def read_and_process(data_path):
    df = pd.read_csv(data_path)
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    date_time = df.pop('Date Time')

    df['wv (m/s)'] = df['wv (m/s)'].replace(-9999.0, 0.0)
    df['max. wv (m/s)'] = df['max. wv (m/s)'].replace(-9999.0, 0.0)

    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')
    wd_rad = df.pop('wd (deg)') * np.pi / 180
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)

    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    n = len(df)
    train_df = df[:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    scaler = StandardScaler()
    scaler.fit(train_df)
    train_data = scaler.transform(train_df)
    val_data = scaler.transform(val_df)
    test_data = scaler.transform(test_df)

    return df, train_data, val_data, test_data, scaler


# ------------------------------
# 2. 时间窗口 Dataset 类
# ------------------------------
class TempDataset(Dataset):
    def __init__(self, data, input_len=24, output_len=12, target_col=1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_len = input_len
        self.output_len = output_len
        self.target_col = target_col
        self.window_size = input_len + output_len

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_len]
        y = self.data[idx + self.input_len: idx + self.window_size, self.target_col]
        return x, y


if __name__ == '__main__':
    df, train_data, val_data, test_data = read_and_process('/root/autodl-tmp/lab/dataset/jena_climate_2009_2016.csv')
    print("数据信息")
    print(df.info())

    rows, columns = df.shape

    if rows > 0 and columns > 0:
        print(f"数据全部加载完成，共{rows}行，{columns}列")
        print(df.head().to_csv(sep='\t', na_rep='nan'))
    else:
        print("数据加载可能存在问题，行数或列数为0")
        
    print(f"\n训练集形状: {train_data.shape}")
    print(f"验证集形状: {val_data.shape}")
    print(f"测试集形状: {test_data.shape}")