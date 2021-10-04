import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import gzip
import io

from sklearn.linear_model import LinearRegression

weather_data_url = 'https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_station/'


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


def linear_regression(x_col, y_col):
    x = x_col.values.reshape((-1, 1))
    y = y_col.values.reshape((-1, 1))

    regr = LinearRegression().fit(x, y)
    trend = regr.predict(x_col.values.astype(float).reshape((-1, 1)))
    return trend, regr


def plot_trend(data, variable):
    # todo: deviation from trend
    data['trend'] = linear_regression(data['date'], data['value'])

    ax = data.plot(x='date', y='value', color='tab:blue')
    data.plot(x='date', y='trend', color='tab:orange', ax=ax)
    # data.plot(x='date', y='rolling', color='tab:grey', ax=ax)
    # data.plot(x='date', y='trend', color='tab:orange')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{variable} (°F)')


def stat(data):
    data['decade'] = np.floor(data.index.year / 10) * 10
    data['decade'] = data['decade'].astype(int)

    m = data.groupby('decade')['value'].max()
    p = data.groupby('decade')['value'].quantile(.96)
    data['rolling'] = data.value.rolling(window=365 * 5).mean()

    data['year'] = data.set_index('date').index.year
    data['year'] = data['year'].astype(int)
    maximum = data.groupby('year')['value'].max()
    point04 = data.groupby('year')['value'].quantile(.96)
    ave = data.groupby('year')['value'].mean()

    stats = pd.DataFrame({'max': maximum,
                          'quant_96': point04,
                          'ave': ave}).reset_index()
    stats['max_trend'] = linear_regression(stats['year'], stats['max'])
    stats['quant_trend'] = linear_regression(stats['year'], stats['quant_96'])
    stats['ave_trend'] = linear_regression(stats['year'], stats['ave'])

    ax = stats.plot(x='year', y='quant_96', color='tab:blue')
    # stats.plot(x='year', y='max', color='tab:orange', ax=ax)
    # stats.plot(x='year', y='ave', color='tab:grey', ax=ax)

    stats.plot(x='year', y='quant_trend', color='tab:blue', linestyle='dotted', ax=ax)
    # stats.plot(x='year', y='max_trend', color='tab:orange', linestyle='dotted', ax=ax)
    # stats.plot(x='year', y='ave_trend', color='tab:grey', linestyle='-', ax=ax)

    return m, p


if __name__ == '__main__':
    phila = 'USW00013739'
    la = 'USW00023174'
    nyc = 'USW00013739'
    chi = 'USW00094846'
    dallas = 'USW00003927'
    sea_tac = 'USW00024233'
    spokane = 'USW00024157'
    pitt = 'USW00094823'
    sf = 'USW00023234'

    d = get_weather(sea_tac)

    tmax = d.loc[d['variable'] == 'TMAX'].copy()
    tmin = d.loc[d['variable'] == 'TMIN'].copy()

    tmax['year'] = tmax.set_index('date').index.year
    tmax['year'] = tmax['year'].astype(int)
    point04 = tmax.groupby('year')['value'].quantile(1)
    point04 = point04.reset_index()

    point04['quant_trend'], trend = linear_regression(point04['year'], point04['value'])

    std_dev = point04['value'].std()
    mu = 0
    num_years = 30

    noise = np.random.normal(mu, std_dev, num_years)
    future_years = np.arange(2022, 2052)
    future_trend = trend.predict(future_years.reshape(-1, 1)).reshape(-1)
    future = pd.DataFrame({'future_trend': future_trend,
                           'future_96vals': future_trend + noise},
                          index=future_years)
    future.loc[2021, 'future_trend'] = point04['quant_trend'].iloc[-1]
    future.loc[2021, 'future_96vals'] = point04['value'].iloc[-1]
    future = future.sort_index()

    plt.figure()
    plt.plot(point04['year'], point04['value'], label='Actual Annual Values',
             color='tab:blue')
    plt.plot(point04['year'], point04['quant_trend'], label='Actual Trend',
             color='tab:orange')

    plt.plot(future['future_96vals'], color='tab:grey',
             label='Possible Future Annual Values')
    plt.plot(future['future_trend'], linestyle='dotted',
             color='tab:grey', label='Historic Trend Projected')
    plt.ylabel('Temperature (°F)')
    plt.title('Annual Maximum Dry Bulb Temperature - Seattle, WA')
    plt.legend()
