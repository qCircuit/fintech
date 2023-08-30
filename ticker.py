import numpy as np
import pandas as pd
import yfinance as yf

from scipy.signal import argrelextrema

import configur
import util



class Ticker:
    def __init__(self, symbol:str):
        self.symbol = symbol
        self.data = pd.DataFrame(dtype=object, columns=['Open','High','Low','Close','Adj Close','Volume'])
    
    def yf_download(self, start_date: str, end_date: str, interval: str):
        self.data = yf.download(
            tickers=self.symbol, 
            start=start_date, 
            end=end_date, 
            interval=interval
        )   

        path = f"{configur.data_path}{self.symbol}_{start_date}_{end_date}.parquet"
        self.data.to_parquet(path, index=False)
        util.logger.info(f"Data downloaded and saved at {path}")

        return self.data
    
    def upload_data(self, start_date: str, end_date: str):
        path = f"{configur.data_path}{self.symbol}_{start_date}_{end_date}.parquet"
        self.data = pd.read_parquet(path)
        util.logger.info(f"data loaded: {start_date} - {end_date}, shape: {self.data.shape}")

        return self.data
    
    def get_smooth(self, series_name:str=configur.target, smoothing_factor:float=0.3):
        # smooth the series with the defined factor
        smoothed_series = self.data[series_name].ewm(alpha=smoothing_factor).mean()
        # append to the data
        self.data["smoothed_series"] = smoothed_series
        util.logger.info(f"data smoothed, shape: {self.data.shape}")

        return smoothed_series
    
    def get_extrema(self) -> tuple:
        # get local minima and maxima of the smoothed series

        smoothed_series = self.data["smoothed_series"]
        # get extrema
        local_min_idx = argrelextrema(smoothed_series.values, np.less)
        local_max_idx = argrelextrema(smoothed_series.values, np.greater)

        # drop extrema if the distance to the previos ones <= 2 periods
        local_min_idx = np.delete(
            local_min_idx[0], 
            np.argwhere(np.diff(local_min_idx[0]) <= 2) + 1
        )
        local_max_idx = np.delete(
            local_max_idx[0], 
            np.argwhere(np.diff(local_max_idx[0]) <= 2) + 1
        )

        # append to the data
        self.data["local_min"] = smoothed_series[local_min_idx].combine_first(
            pd.Series(0, index=range(len(smoothed_series)), name="local_min")
        ) 
        self.data["local_max"] = smoothed_series[local_max_idx].combine_first(
            pd.Series(0, index=range(len(smoothed_series)), name="local_max")
        )
        util.logger.info(f"extrema collected, shape: {self.data.shape}")

        return (local_min_idx, local_max_idx)
    
    def get_support(self):

        local_min_idx, _ = self.get_extrema()

        # append start / end points
        local_min_idx = np.insert(local_min_idx, 0, 0)
        local_min_idx = np.append(local_min_idx, self.data.index[-1])

        # get support 
        support_line = list()
        for i in range(len(local_min_idx)-1):
            support_line.append(util.get_line(
                local_min_idx[i], 
                local_min_idx[i+1],
                *util.get_coefs(
                    local_min_idx[i],
                    local_min_idx[i+1],
                    self.data["smoothed_series"][local_min_idx[i]],
                    self.data["smoothed_series"][local_min_idx[i+1]]
                )
            ))
        support_line = np.array([x for sl in support_line for x in sl])

        # append to the data
        support_line = pd.Series(
            support_line[:,1], index=support_line[:,0].astype(int), name="support_line"
        )
        support_line = support_line[~support_line.index.duplicated()]
        self.data["support_line"]=support_line.combine_first(
            pd.Series(0, index=self.data.index, name="support_line")
        )
        util.logger.info(f"support lines collected, shape: {self.data.shape}")

        return support_line
    
    def get_resistance(self):

        _, local_max_idx = self.get_extrema()

        # append start / end points
        local_max_idx = np.insert(local_max_idx, 0, 0)
        local_max_idx = np.append(local_max_idx, self.data.index[-1])

        # get resistance

        resistance_line = list()
        for i in range(len(local_max_idx)-1):
            resistance_line.append(util.get_line(
                local_max_idx[i], 
                local_max_idx[i+1],
                *util.get_coefs(
                    local_max_idx[i],
                    local_max_idx[i+1],
                    self.data["smoothed_series"][local_max_idx[i]],
                    self.data["smoothed_series"][local_max_idx[i+1]]
                )
            ))
        resistance_line = np.array([x for sl in resistance_line for x in sl])

        # append to the data
        resistance_line = pd.Series(
            resistance_line[:,1], index=resistance_line[:,0].astype(int), name="resistance_line"
        )
        resistance_line = resistance_line[~resistance_line.index.duplicated()]
        self.data["resistance_line"]=resistance_line.combine_first(
            pd.Series(0, index=self.data.index, name="resistance_line")
        )
        util.logger.info(f"resistance lines collected, shape: {self.data.shape}")

        return resistance_line
    
    def get_moving_average(self):
        # get moving average of short and long periods

        self.get_smooth()

        moving_average_short = self.data["smoothed_series"].rolling(configur.ma_short_period).mean()
        moving_average_long = self.data["smoothed_series"].rolling(configur.ma_long_period).mean()

        moving_average_short = moving_average_short.fillna(0)
        moving_average_long = moving_average_long.fillna(0)

        self.data["moving_average_short"] = moving_average_short
        self.data["moving_average_long"] = moving_average_long

        util.logger.info(f"moving averages collected, shape: {self.data.shape}")

        return moving_average_short, moving_average_long
    
    def get_stochastic_oscillator(self, period):
        self.get_smooth()
        
        # calc components
        current = self.data.smoothed_series.iloc[period:]
        lowest = [self.data.smoothed_series.iloc[i-period:i+period].min() for i in range(period, len(self.data.smoothed_series))]
        highest = [self.data.smoothed_series.iloc[i-period:i+period].max() for i in range(period, len(self.data.smoothed_series))]
        nominator = [c-l for c,l in zip(current, lowest)]
        denominator = [h-l for h,l in zip(highest, lowest)]

        # get oscillator values
        stochastic_oscillator = pd.Series(
            [n/d for n,d in zip(nominator, denominator)],
            name='stochastic_oscillator'
        )

        # add zeros to the incalculable start period
        stochastic_oscillator = pd.concat([
            pd.Series([0] * period),
            stochastic_oscillator
        ]).reset_index(drop=True)

        self.data["stochastic_oscillator"] = stochastic_oscillator
        util.logger.info(f"stochastic oscillator collected, shape: {self.data.shape}")

        return stochastic_oscillator

        
