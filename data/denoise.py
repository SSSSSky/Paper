'''
Denoise 去噪
'''
import matplotlib.pyplot as plt
import pywt
from statsmodels.tsa.seasonal import seasonal_decompose

from data_api import *


class Denoise():
    def __init__(self, datasets):
        self.datasets = datasets

    def wavelet(self):
        '''小波变换'''
        self.datasets.reset_index(inplace=True)
        self.datasets['Trddt'] = pd.to_datetime(self.datasets['Trddt'])
        self.datasets = self.datasets.set_index('Trddt')
        cA, cD = pywt.dwt(self.datasets.Clsprc, 'db1')
        print(cA, cD)

    def tsd(self):
        '''时间序列分解
        '''
        self.datasets.reset_index(inplace=True)
        self.datasets['Trddt'] = pd.to_datetime(self.datasets['Trddt'])
        self.datasets = self.datasets.set_index('Trddt')
        decomposition = seasonal_decompose(self.datasets.Clsprc, freq=20)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        plt.figure(4, (16, 9))
        plt.subplot(411)
        plt.plot(self.datasets.Clsprc)
        plt.subplot(412)
        plt.plot(trend)
        plt.subplot(413)
        plt.plot(seasonal)
        plt.subplot(414)
        plt.plot(residual)
        plt.show()

def main():
    df = get_price(['000001'], start_date='2017-01-01', end_date='2017-12-31')
    print(df.head())
    denoise = Denoise(df)
    # denoise.tsd()
    denoise.wavelet()

if __name__ == '__main__':
    main()
