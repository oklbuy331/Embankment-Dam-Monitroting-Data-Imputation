import os
from typing import Optional, Sequence, List

import numpy as np
import pandas as pd
from tsl import logger
from datetime import datetime

from tsl.data.datamodule.splitters import AtTimeStepSplitter
from tsl.ops.similarities import gaussian_kernel
from tsl.datasets.prototypes import PandasDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.data.utils import HORIZON
from sklearn.feature_selection import mutual_info_regression


class PoreWaterPressure(PandasDataset, MissingValuesMixin):
    """

    """

    similarity_options = {'distance', 'Pearson', 'mutual_information', 'Granger'}
    temporal_aggregation_options = {'mean', 'nearest'}
    spatial_aggregation_options = {'mean'}

    def __init__(self, root: str = None,
                 similarity_score: str = 'distance',
                 test_months: Sequence = None,
                 impute_nans: bool = True,
                 infer_eval_from: str = 'next',
                 freq: Optional[str] = None,
                 masked_sensors: Optional[Sequence] = None):
        self.root = root
        self.test_months = test_months
        self.infer_eval_from = infer_eval_from  # [next, previous]
        if masked_sensors is None:
            self.masked_sensors = []
        else:
            self.masked_sensors = list(masked_sensors)
        df, mask, eval_mask, dist = self.load(impute_nans=impute_nans)
        super().__init__(dataframe=df,
                         attributes=dict(dist=dist),
                         mask=mask,
                         freq=freq,
                         similarity_score=similarity_score,
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         default_splitting_method='pwp',
                         name='PWP')
        self.set_eval_mask(eval_mask)

    @property
    def raw_file_names(self) -> List[str]:
        return ['data.csv']

    @property
    def required_file_names(self) -> List[str]:
        return self.raw_file_names + ['pwp_dist.npy']

    def load_raw(self):
        path = os.getcwd() + '/spin/datasets/data.csv'
        dist = np.load(os.getcwd() + '/spin/datasets/dist.npy')
        eval_mask = None
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        return pd.DataFrame(df), eval_mask, dist

    def load(self, impute_nans=True):
        # load readings and stations metadata
        df, eval_mask, dist = self.load_raw()
        # compute the masks:
        mask = (~pd.isna(df.values)).astype('uint8')  # 1 if value is valid
        if eval_mask is None:
            eval_mask = self.infer_mask(df, infer_from=self.infer_eval_from)
        # 1 if value is ground-truth for imputation
        eval_mask = eval_mask.astype('uint8')
        if len(self.masked_sensors):
            eval_mask[:, self.masked_sensors] = mask[:, self.masked_sensors]
        return df, mask, eval_mask, dist

    def get_splitter(self, method: Optional[str] = None, **kwargs):
        if method == 'pwp':
            first_val_ts = datetime.strptime("2020-04-16", "%Y-%m-%d")
            first_test_ts = datetime.strptime("2020-05-16", "%Y-%m-%d")
            return AtTimeStepSplitter(first_val_ts=first_val_ts,
                                      first_test_ts=first_test_ts)

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            # calculate similarity in terms of spatial distance
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)

        elif method == "Pearson":
            # calculate similarity in terms of Pearson correlation
            return self.df.corr().values

        elif method == "mutual_information":
            # calculate similarity in terms of mutual information between sensor records
            mi = np.load(os.getcwd() + '\\spin\\datasets\\mi.npy')
            for i in range(len(mi)):
                min = mi[i].min(); max = mi[i].max()
                mi[i] = (mi[i] - min)/(max - min)
            return mi

        else:
            # calculate similarity in terms of Granger causalty
            theta = np.std(self.dist[:36, :36])
            return gaussian_kernel(self.dist, theta=theta)

    def infer_mask(self, df, infer_from='next'):
        """Infer evaluation mask from DataFrame. In the evaluation mask a value is 1
        if it is present in the DataFrame and absent in the :obj:`infer_from` month.

        Args:
            df (pd.Dataframe): The DataFrame.
            infer_from (str): Denotes from which month the evaluation value must be
                inferred. Can be either :obj:`previous` or :obj:`next`.

        Returns:
            pd.DataFrame: The evaluation mask for the DataFrame.
        """
        mask = (~df.isna()).astype('uint8')
        eval_mask = pd.DataFrame(index=mask.index, columns=mask.columns,
                                 data=0).astype('uint8')
        if infer_from == 'previous':
            offset = -1
        elif infer_from == 'next':
            offset = 1
        else:
            raise ValueError('`infer_from` can only be one of {}'.format(['previous', 'next']))
        months = sorted(set(zip(mask.index.year, mask.index.month)))
        length = len(months)
        for i in range(length):
            j = (i + offset) % length
            year_i, month_i = months[i]
            year_j, month_j = months[j]
            cond_j = (mask.index.year == year_j) & (mask.index.month == month_j)
            mask_j = mask[cond_j]
            offset_i = 12 * (year_i - year_j) + (month_i - month_j)
            mask_i = mask_j.shift(1, pd.DateOffset(months=offset_i))
            mask_i = mask_i[~mask_i.index.duplicated(keep='first')]
            mask_i = mask_i[np.in1d(mask_i.index, mask.index)]
            i_idx = mask_i.index
            eval_mask.loc[i_idx] = ~mask_i.loc[i_idx] & mask.loc[i_idx]
        return eval_mask
