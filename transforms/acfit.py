import copy

import numpy as np
import pandas as pd

from gluonts.dataset.field_names import FieldName
from gluonts.transform import SimpleTransformation

from statsmodels.tsa.stattools import acf

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Define acf transform
class ACFITransform(SimpleTransformation):
    def __init__(self, nlags, freq, name='no-name'):
        super().__init__()
        self.nlags = nlags
        self.freq = freq
        self.name = name
    def transform(self, data):
        tf_data = copy.deepcopy(data)
        targets = [item[FieldName.TARGET] for item in tf_data]
        starts = [item[FieldName.START] for item in tf_data]
        for i,target in enumerate(targets):
            start = starts[i]+pd.Timedelta(self.nlags, unit=self.freq)
            acf_weights = acf(target, nlags=self.nlags)
            #acf_weights[acf_weights<0] = 0
            # Need positivity and confidence bounds
            target = np.convolve(target, acf_weights, mode='valid')
            tf_data[i][FieldName.TARGET] = target
            tf_data[i][FieldName.START] = start
            if i>0:
                print('Warning: Multiple time series detected. Only the first acf is stored.')
            else:
                self.acf = acf_weights
                self.start = targets[i][:len(self.acf)-1]
                plt.figure()
                plot_acf(target, lags=self.nlags)
                plt.savefig(f'./out/{self.name}_acfit_{self.freq}_weights.png')
        return tf_data
    def invert(self, dataset, samples=None):
        data = dataset[0][FieldName.TARGET]
        inv_data = np.concatenate((self.start, np.zeros_like(data)))
        for i in range(len(self.acf), len(inv_data)):
            inv_data[i] = (data[i - len(self.acf)] - np.dot(self.acf[1:][::-1], inv_data[i-len(self.acf)+1:i])) / self.acf[0]
        dataset[0][FieldName.TARGET] = inv_data
        return dataset
