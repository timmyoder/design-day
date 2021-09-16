import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import gzip
import io

from sklearn.linear_model import LinearRegression

weather_data_url = 'https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_station/'


def generate_noise(base_signal):
    """ function to add noise of a given probability to an input signal"""
    return base_signal


def climate_indicator():
    """generate the underlying signal for the climate indicator, relatively
    stable until increase in future"""
    x = np.arange(0, 1, .01)
    y = np.tan(x)
    return x, y


def plot_indicator(signal):
    """plotting functions for climate signal"""
    plt.plot(signal[0], signal[1])


def parse_hnc_content(raw_content):
    columns = ['station', 'date', 'variable', 'value',
               'MFLAG1', 'QFLAG1', 'SFLAG1', 'unknown']

    io_stream = io.StringIO(raw_content.decode())

    parsed_content = pd.read_csv(filepath_or_buffer=io_stream,
                                 header=None,
                                 names=columns,
                                 dtype={'station': object,
                                        'date': object,
                                        'variable': object,
                                        'value': int,
                                        'MFLAG1': object,
                                        'QFLAG1': object,
                                        'SFLAG1': object,
                                        'unknown': object})

    # keep only the tmax and tmin variables
    element_keep = ['TMAX', 'TMIN']
    parsed_content = parsed_content[parsed_content['variable'].isin(element_keep)]

    # remove values that failed quality checks
    parsed_content = parsed_content[parsed_content['QFLAG1'].isna()]

    parsed_content['date'] = pd.to_datetime(parsed_content['date'],
                                            format='%Y%m%d')
    # convert from tenths of C to F
    parsed_content['value'] = parsed_content['value'] * 9 / 50 + 32

    parsed_content.drop(columns=['station',
                                 'MFLAG1', 'QFLAG1',
                                 'SFLAG1', 'unknown'],
                        inplace=True)

    return parsed_content


def get_weather(station):
    path = f'{weather_data_url}{station}.csv.gz'
    compressed_content = requests.get(path)
    raw_content = gzip.decompress(compressed_content.content)

    return parse_hnc_content(raw_content)


def plot_trend(data, variable):
    # todo: deviation from trend
    x = data['date'].values.reshape((-1, 1))
    y = data['value'].values.reshape((-1, 1))

    regr = LinearRegression().fit(x, y)
    trend = regr.predict(data['date'].values.astype(float).reshape((-1, 1)))
    data['trend'] = trend

    ax = data.plot(x='date', y='value', color='tab:blue')
    data.plot(x='date', y='trend', color='tab:orange', ax=ax)
    # data.plot(x='date', y='trend', color='tab:orange')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{variable} (°F)')


if __name__ == '__main__':
    phila = 'USW00013739'
    d = get_weather(phila)

    tmax = d.loc[d['variable'] == 'TMAX']
    tmin = d.loc[d['variable'] == 'TMIN']

    plot_trend(tmax, 'Daily Max Temp')
    # plot_trend(tmin, 'Daily Min Temp')

    # indicator = climate_indicator()
    # noisy_indicator = generate_noise(indicator)
    # plot_indicator(indicator)
