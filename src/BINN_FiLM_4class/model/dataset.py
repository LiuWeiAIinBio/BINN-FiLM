import pandas as pd
import torch
from typing import List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class BINNDataset(Dataset):
    def __init__(self, X: pd.DataFrame, T: pd.DataFrame, y: pd.DataFrame):
        """
        X: a DataFrame that contains sample data whose shape is (sample size * features size)
        y: a DataFrame that contains sample label data whose shape is (sample size * labels size)
        """
        self.data = self.get_data(X, T, y)

    def __getitem__(self, index):
        x_batch, t_batch, y_batch = self.data[index]
        x_batch = torch.tensor(x_batch, dtype=torch.float32)
        t_batch = torch.tensor(t_batch, dtype=torch.float32)
        y_batch = torch.tensor(y_batch, dtype=torch.long)
        return x_batch, t_batch, y_batch

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data(X, T, y) -> list[tuple[Any, Any, Any]]:
        data = []

        for i in range(X.shape[0]):
            x_i = X.iloc[i].values
            t_i = T.iloc[i].values.squeeze()
            y_i = y.iloc[i].values.squeeze()
            data.append((x_i, t_i, y_i))

        return data
