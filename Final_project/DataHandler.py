import pandas as pd

class DataHandler:
    """
    DataHandler - class for preprocessing data.
    
    """
    def __init__(self, name):
        self.name = name


    def readFile(self):
        path = 'Data/' + self.name + '.csv'
        #import
        data = pd.read_csv(path)
        # drop unnecessary columns
        data = data.drop(columns = ['time', 'atotal'])
        # moving average + drop NaN due to filtering
        data = data.rolling(window=15).mean().dropna()
        # convert dataframe into array
        return data.values