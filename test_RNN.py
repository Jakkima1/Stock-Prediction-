def main():

    # Importing Libraries
    import pandas as pd
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error

    from keras.models import load_model

    # Importing test_data_RNN.csv file

    test_Load_dataset = pd.read_csv('data/test_data_RNN.csv')

    X_test = test_Load_dataset.iloc[:, 0:12].values

    Y_test = test_Load_dataset.iloc[:, -1].values
    Y_test = np.array(Y_test).reshape(-1, 1)

    # Loading saved model from models folder

    saved_model_RNN = load_model("./models/RNN_model.model")

    # Data Normalization
    sc = MinMaxScaler(feature_range=(0, 1))
    X_test = sc.fit_transform(X_test)
    Y_test = sc.fit_transform(Y_test)

    # Predicting the test data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    test_data_predict_model = saved_model_RNN.predict(X_test)

    # Finding RMSE metrics
    final_loss = math.sqrt(mean_squared_error(Y_test, test_data_predict_model))
    print('Test data loss:', final_loss)

    # Plotting the true and predicted values
    plt.rcParams["figure.figsize"] = (30, 6)
    plt.plot(Y_test, color='red', label='true values')
    plt.plot(test_data_predict_model, color='grey', label='predicted values')
    plt.title('Stock Prediction')
    plt.legend()
    plt.show()





if __name__ == "__main__":
    main()
