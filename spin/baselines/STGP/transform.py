import os
import numpy as np
import pandas as pd

# Import spatial temporal Gaussian process class
from STGP import STGP


def form_df(df, sensor, coordinates):

    long = coordinates[:, 0]
    lat = coordinates[:, 1]
    alt = coordinates[:, 2]

    data_df = pd.DataFrame()
    for i in range(len(df)):
        date = np.array([(df.index[i] - df.index[0]).days])[:].repeat(20, axis=0)
        level = df.iloc[i, :].values
        data = dict(sensor=sensor, long=long, lat=lat, alt=alt, date=date, level=level)
        data_df_i = pd.DataFrame(data)
        data_df = pd.concat([data_df, data_df_i], ignore_index=True)

    return data_df


if __name__ == "__main__":

    # Load data
    filename = os.path.join(os.getcwd(), 'data.csv')
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)

    sensor = np.array(['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12',
                       'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20'])

    coordinates = np.array([[-12, 1280, 0], [-3.5, 1280, 0], [8, 1280, 0],
                            [-12, 1290, 0], [-3.5, 1290, 0], [8.5, 1290, 0],
                            [-12, 1305, 0], [-3.5, 1305, 0], [6.5, 1305, 0],
                            [-12, 1280, 55], [-3.5, 1280, 55], [8, 1280, 55],
                            [-12, 1290, 55], [-3.5, 1290, 55], [8.5, 1290, 55],
                            [-12, 1305, 55], [-3.5, 1305, 55], [6.5, 1305, 55],
                            [-2.5, 1265, 0], [2.5, 1265, 0]], dtype='float64')

    # Set temporal parameters
    target_var = 'level'
    target_name = "Seepage Level"
    cov_func_name = "Squared exponential"

    # Form the Dataframe
    df = form_df(df, sensor, coordinates)
    long_low = df['long'].min()
    long_up = df['long'].max()
    lat_low = df['lat'].min()
    lat_up = df['lat'].max()
    alt_low = df['alt'].min()
    alt_up = df['alt'].max()
    date_low = df['date'].min()

    # Instantiate STGP using data
    stgp = STGP(df, target_var, target_name, long_low, long_up, lat_low, lat_up,
                alt_low, alt_up, date_low, cov_func_name)

    # Make bounds for optimiser (log-scale) - Could try to find accuracy of sensors
    # bounds = ((np.log(0.5 * stgp.hyper_params[0]), np.log(2 * stgp.hyper_params[0])),
    #           (np.log(0.5 * stgp.hyper_params[1]), np.log(2 * stgp.hyper_params[1])),
    #           (np.log(0.5 * stgp.hyper_params[2]), np.log(2 * stgp.hyper_params[2])),
    #           (np.log(0.5 * stgp.sigma_n), np.log(2 * stgp.sigma_n)))

    bounds = ((-2, 2),
              (-2, 2),
              (-2, 2),
              (-2, 2))

    # bounds = None
    # Try to fit it - giving a long time-lengthscale so not changing much over time
    stgp.fit(bounds=bounds, verbose=True)

    X_star, _ = stgp.sampleX(size=2000)
    prediction = stgp.predict(X_star)
    stgp.plot_preds(save=True)
