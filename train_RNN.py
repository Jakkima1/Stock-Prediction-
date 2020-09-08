# #import required packages
def main():
    # Importing Libraries
    import pandas as pd
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    # Sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error

    # Create the Stacked LSTM model
    from keras.models import Sequential
    from keras.layers import LSTM, Dropout, Dense, Activation

    # Loading the dataset
    # train_data = pd.read_csv(r'./data/q2_dataset.csv')
    #
    # # Selecting volume, open, High aand low columns
    # train_set = train_data.iloc[:, 2:6]
    # # Storing the dates in dates_data from the dataset
    # dates_data = train_data.iloc[:, 0:1]
    #
    # # selecting volume column and reshaping
    # data1 = train_data[train_data.columns[2]]
    # data1 = np.array(data1).reshape(-1, 1)
    # # selecting open column and reshaping
    # data2 = train_data[train_data.columns[3]]
    # data2 = np.array(data2).reshape(-1, 1)
    # # selecting high column and reshaping
    # data3 = train_data[train_data.columns[4]]
    # data3 = np.array(data3).reshape(-1, 1)
    # # selecting low column and reshaping
    # data4 = train_data[train_data.columns[5]]
    # data4 = np.array(data4).reshape(-1, 1)
    #
    # # Function selecting past 3 days data for each column(volume, open, high, low)
    # def prepare_dataset(data, data_step=1):
    #     x_data, y_data = [], []
    #     for k in range(len(data) - data_step):
    #         b = data[k:(k + data_step), 0]  # k = 0, 0,1,2   3
    #         x_data.append(b)
    #         y_data.append(data[k + data_step, 0])
    #     return np.array(x_data), np.array(y_data)
    #
    # data_step = 3
    # # Calling prepare_dataset function for Volume
    # volume, y_train = prepare_dataset(data1, data_step)
    # # Calling prepare_dataset function for open
    # opn, label = prepare_dataset(data2, data_step)
    # # Calling prepare_dataset function for high
    # high, y_train = prepare_dataset(data3, data_step)
    # # Calling prepare_dataset function for low
    # low, y_train = prepare_dataset(data4, data_step)
    #
    # # I have allocated features as volume1, volume2, volume3, open1, open2, open3, high1, high2, high3, low1, low2, low3.
    # # Thus I have got 12 features as input for new dataset.
    # train = np.concatenate((volume, opn, high, low), axis=1)
    # train = train[0:]
    # train_data_store = pd.DataFrame(train, columns=['volume1', 'volume2', 'volume3', 'open1', 'open2', 'open3', 'high1',
    #                                                 'high2', 'high3', 'low1', 'low2', 'low3'])
    # # Appending dates column
    # train_data_store.insert(0, 'dates', dates_data[1:1257])
    # # open column as the target
    # target = np.array(data2).reshape(-1, 1)
    # target = target[0:1256]
    #
    # # Randomize and splitting the data into train and validation(70% and 30%)
    # X_train, X_test, Y_train, Y_test = train_test_split(train, target, test_size=0.3, random_state=42)
    #
    # # Exporting Train and Test data into .csv files
    # # Converting X_train and Y_train data into .csv
    # X_Train = pd.DataFrame(X_train)
    #
    # Y_Train = pd.DataFrame(Y_train)
    #
    # Train_data_csv = pd.concat([X_Train, Y_Train], axis=1)
    #
    # Train_data_csv.to_csv(r'data\train_data_RNN.csv', index=False)
    #
    # # Converting X_test and Y_test data into .csv
    #
    # X_Test = pd.DataFrame(X_test)
    #
    # Y_Test = pd.DataFrame(Y_test)
    #
    # Test_data_csv = pd.concat([X_Test, Y_Test], axis=1)
    #
    # Test_data_csv.to_csv(r'data\test_data_RNN.csv', index=False)

    # Loading train_data_RNN.csv file

    train_Load_data = pd.read_csv('data/train_data_RNN.csv')

    X_train=train_Load_data.iloc[:, 0:12].values

    Y_train=train_Load_data.iloc[:,-1].values

    Y_train = np.array(Y_train).reshape(-1, 1)

    # Data Preprocessing for train data
    # Data Normalization for train data
    sc = MinMaxScaler(feature_range=(0, 1))
    X_train = sc.fit_transform(X_train)
    Y_train = sc.fit_transform(Y_train)

    # Reshaping the train data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Building neural network
    model = Sequential()

    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.35))

    model.add(LSTM(100, return_sequences=True, activation='sigmoid'))
    model.add(Dropout(0.45))

    model.add(LSTM(100, activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.summary()

    model.fit(X_train, Y_train, epochs=10, batch_size=64)
    # Saving trained Model
    model.save("./models/RNN_model.model")


if __name__ == "__main__":
    main()
	
