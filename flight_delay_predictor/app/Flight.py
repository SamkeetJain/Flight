import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression


def predict(a,m,c,w):

    # Load Data
    dataFrame = pd.read_csv('static/data.csv',header=None)
    columns = ['year','month','carrier','carrier_name','airport','airport_name','arr_flights','arr_del15','carrier_ct','weather_ct','nas_ct','security_ct','late_aircraft_ct','arr_cancelled','arr_diverted','arr_delay','carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay','temp']
    dataFrame.columns = columns

    weather_data = pd.read_csv('static/weather_data.csv',header=None)
    weather_data.columns = ['weather']

    dataFrame = dataFrame[['year','carrier','month','airport','weather_ct']]
    weather_data = weather_data[['weather']].astype(np.float)

    X_0 = dataFrame.iloc[:,0:1]
    X_1 = dataFrame.iloc[:,1:2]
    X_2 = dataFrame.iloc[:,2:3]
    X_3 = dataFrame.iloc[:,3:4]
    X_4 = weather_data.iloc[:,0:1]

    Y = dataFrame.iloc[:,4].values


    # Categorical columns for Carrier
    label_encoder_X_1 = LabelEncoder()
    X_1 = X_1.apply(label_encoder_X_1.fit_transform)
    onehotencoder_1 = OneHotEncoder()
    onehotencoder_1.fit(X_1)
    X_1 = onehotencoder_1.transform(X_1).toarray()

    labels_1 = label_encoder_X_1.classes_
    mappings_1 = {}
    for index, label in zip(range(len(labels_1)), labels_1):
        mappings_1[label]=index


    # Categorical columns for month
    label_encoder_X_2 = LabelEncoder()
    X_2 = X_2.apply(label_encoder_X_2.fit_transform)
    onehotencoder_2 = OneHotEncoder()
    onehotencoder_2.fit(X_2)
    X_2 = onehotencoder_2.transform(X_2).toarray()

    labels_2 = label_encoder_X_2.classes_
    mappings_2 = {}
    for index, label in zip(range(len(labels_2)), labels_2):
        mappings_2[label]=index


    # Categorical columns for Airport code
    label_encoder_X_3 = LabelEncoder()
    X_3 = X_3.apply(label_encoder_X_3.fit_transform)
    onehotencoder_3 = OneHotEncoder()
    onehotencoder_3.fit(X_3)
    X_3 = onehotencoder_3.transform(X_3).toarray()

    labels_3 = label_encoder_X_3.classes_
    mappings_3 = {}
    for index, label in zip(range(len(labels_3)), labels_3):
        mappings_3[label]=index


    # Check for NAN values
    X_1[np.isnan(X_1)] = np.median(X_1[~np.isnan(X_1)])
    X_2[np.isnan(X_2)] = np.median(X_2[~np.isnan(X_2)])
    X_3[np.isnan(X_3)] = np.median(X_3[~np.isnan(X_3)])
    Y[np.isnan(Y)] = np.median(Y[~np.isnan(Y)])

    # Create X dataframe
    data = pd.concat([pd.DataFrame(X_1),pd.DataFrame(X_2),pd.DataFrame(X_3),pd.DataFrame(X_4)],axis=1)
    print(data.shape)

    # Split in to train and test
    X_train,X_test,Y_train,Y_test = train_test_split(data.iloc[:,:].values,Y,test_size=0.2,random_state=0)

    model = LinearRegression()
    model.fit(X_train,Y_train)

    # bind user input
    airport,month,carrier,weather = a,m,c,w

    # 26 + 12 + 386 + 1
    # Generate X predict column
    carrier_column = mappings_1[carrier]
    month_column = 26 + mappings_2[int(month)]
    airport_column = 26 + 12 + mappings_3[airport]
    weather_data_column = 386 + 26 + 12

    X_predict = [0 for _ in range(425)]
    X_predict[carrier_column] = 1
    X_predict[month_column] = 1
    X_predict[airport_column] = 1
    X_predict[weather_data_column] = int(weather)

    # Finally predict
    y_predict = model.predict([X_predict])
    return -y_predict if y_predict < 0 else y_predict