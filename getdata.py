import pandas_datareader.data as web
import pandas_datareader.nasdaq_trader as ndq
import pandas as pd
import numpy as np 
import yfinance as yfin

COMPO_CAC40 = "Compo_CAC40.csv"


def get_tickers_data(tickers, start_date, end_date, 
                       clean_it=True, returns_only=True, cumreturns_only=False):
    if isinstance(tickers, str):
        tickers = [tickers]
    assert isinstance(tickers, list)
    yfin.pdr_override()
    data = web.get_data_yahoo(tickers, start_date, end_date)
    if clean_it: # Keep only close data
        data = clean_data(data)
        if returns_only:
            data = get_returns(data, cumulative=cumreturns_only)[1:]
    return data

def get_returns(data:pd.DataFrame, cumulative=False):
    cumreturns_data = data.copy()
    for col in cumreturns_data.columns:
        new_serie = np.log(cumreturns_data[col] / cumreturns_data[col].shift(1))
        cumreturns_data[col] = new_serie.cumsum() if cumulative else new_serie

    return cumreturns_data


def clean_data(data, columns=["Close"]):
    if len(columns)==1:
        # Get the relevant columns only
        relevant_data = data[columns[0]]
        relevant_data = pd.DataFrame(relevant_data)
    elif len(columns)>1:
        relevant_data = data[columns]
    else:
        return False
    # Get all the weekdays
    first_date = relevant_data.index[0]
    last_date = relevant_data.index[-1]
    all_weekdays = pd.date_range(start=first_date, end=last_date, freq='B')
    # Put the all weekdays as index 
    relevant_data = relevant_data.reindex(all_weekdays)
    relevant_data = relevant_data.fillna(method='bfill')
    return relevant_data
    

def get_NASDAQ_tickers():
    yfin.pdr_override()
    return ndq.get_nasdaq_symbols()['Nasdaq Traded'].index


def get_CAC40_compo():
    compo = pd.read_csv(COMPO_CAC40,delimiter=",")
    return compo[['Ticker', 'Name', 'Weighting']]


def get_CAC40_tickers(nb=None):
    full = list(get_CAC40_compo().Ticker)
    if nb is None:
        return full
    else:
        if nb >= len(full):
            return full
        else:
            return full[:nb]


def get_custom_CAC40(data:pd.DataFrame, weights:np.ndarray):
    data_all = data.copy()
    data_all = data_all.values
    assert (weights.shape[0]==data_all.shape[1] and weights.shape[1]==1)
    custom_index_val = data_all @ weights
    return pd.DataFrame({"Custom_CAC40":custom_index_val.squeeze()}, index=data.index)


def get_timeserie_cumreturns(ts:pd.DataFrame):
    """Ts should be a one column dataframe with time index

    Parameters
    ----------
    ts : pd.DataFrame
        _description_

    Returns
    -------
    _type_
        _description_
    """
    new_ts = ts.copy()
    assert new_ts.shape[1]==1
    new_ts['returns'] = np.log(new_ts.iloc[:,0] / new_ts.iloc[:,0].shift(1))
    new_ts['cum_returns'] = new_ts['returns'].cumsum()
    return new_ts


if __name__=='__main__':
    import matplotlib.pyplot as plt
        
    # Set the dates
    start_date = '2020-01-02'
    end_date = '2023-02-08'

    # load the data
    all_data = get_tickers_data(get_CAC40_tickers(), start_date, end_date)

    # Reorganized the columns to match the CAC40 compo csv
    all_data = all_data[get_CAC40_tickers()]

    # Get the weights
    weights = np.asarray(get_CAC40_compo().Weighting)[:,None]

    # Get the custom index
    custom_index = get_custom_CAC40(all_data, weights)
    
    # Get the real index
    indx = "^FCHI" #CAC40 id for yahoo
    real_index = get_tickers_data(indx, start_date, end_date)
    real_index.columns=[indx]

    # Plot the custom CAC40 against the real CAC40
    plt.plot(get_timeserie_cumreturns(custom_index)['cum_returns'])
    plt.plot(get_timeserie_cumreturns(real_index)['cum_returns'])
    plt.legend(["Custom CAC40", "Real CAC40"])
    plt.show()

    # Plot the cum returns of all components
    cumreturns = get_returns(all_data, cumulative=True)
    plt.plot(cumreturns)
    plt.show()
    