import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from datetime import datetime as dt
import requests
import matplotlib.pyplot as plt
plt.style.use("ggplot")

class ANN:
    
    def __init__(self, instrument_name, start_date, end_date):
        self.instrument_name = instrument_name
        self.start_date = start_date
        self.end_date = end_date
               
    def getTrades(self):
            try:
                json_df = pd.DataFrame()
                json_data = [1]
    
                startepoch = int((dt.strptime(self.start_date,'%Y-%m-%d').timestamp()-14400)*1000)
                endepoch = int((dt.strptime(self.end_date,'%Y-%m-%d').timestamp()+72000)*1000)
    
                url = 'https://www.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time?' \
                      'instrument_name={0}&end_timestamp={1}&count=1000&' \
                      'include_old=true&sorting=asc&start_timestamp={2}'.format(self.instrument_name,endepoch,startepoch)
    
                json_data = requests.get(url).json()
                json_df = pd.DataFrame(json_data['result']['trades'])
                df = json_df
                #move independent variable to last column, ensures column is in the data set too
                dependent_variable = ['price']
                df = df[[dv for dv in df if dv not in dependent_variable] 
                        + [dv for dv in dependent_variable if dv in df]]
                #convert miliseconds* epoch to datetime (GMT timezone),
                #df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')   
                df = df.drop(['index_price', 'instrument_name', 'liquidation', 'mark_price', 'trade_id', 'trade_seq'], axis = 1) 
                df.to_csv (str(self.start_date)+'_input_data.csv', index = None, header=True)
                return df
               
            except Exception as e:
                print(e)
    
    def data_preprocessing(self, df):
        x = df.iloc[:, :-1].values #independent variable
        y = df.iloc[:, -1].values #dependent variable
        
        #encode direction 
        le = LabelEncoder()
        x[:, 1] = le.fit_transform(x[:,1])  
        
        #data split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, train_size = 0.33, random_state=0)
        
        #normalization
        sc = StandardScaler() 
        x_train_s = sc.fit_transform(x_train)
        x_test_s = sc.transform(x_test) 
        
        return x_train, x_test, x_train_s, x_test_s, y_train, y_test
        
        
    def linear_regression(self):  
        df = self.getTrades()
        x_train, x_test, x_train_s, x_test_s, y_train, y_test = self.data_preprocessing(df)
         
        #XGBoost regressor model
        regressor = xgb.XGBRegressor().fit(x_train_s, y_train)
        y_pred = regressor.predict(x_test_s)
        
        #calculate error
        mse = mean_squared_error(y_test, y_pred)
        #print values side by side
        print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
        print("MSE= "+str(mse))
        return x_train, x_test, x_train_s, x_test_s, y_train, y_test, y_pred, mse

#get data      
DL_model = ANN('BTC-PERPETUAL','2020-05-25','2020-05-25')  
x_train, x_test, x_train_s, x_test_s, y_train, y_test, y_pred, mse = DL_model.linear_regression()

plt.figure(figsize=(12,6), dpi=80)
plt.grid()
plt.xlim(0, 100)
plt.plot(y_pred, label="prediction", color="green", linewidth=2)
plt.plot(y_test, color="blue", linewidth=2, label='real')
plt.title("MSE: %.5f" % mse, fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.legend(loc='best', fontsize=14)
plt.show()

# date = pd.to_datetime(x_test[:,3], unit='ms')


# # How many periods looking back to learn
# n_per_in  = 30

# # How many periods to predict
# n_per_out = 10

# # Features (in this case it's 1 because there is only one feature: price)
# n_features = 1

# # Splitting the data into appropriate sequences
# X, y = split_sequence(list(df.Close), n_per_in, n_per_out)

# # Reshaping the X variable from 2D to 3D
# X = X.reshape((X.shape[0], X.shape[1], n_features))









# # Fitting XGBoost to the Training set
# classifier = XGBClassifier()
# classifier.fit(x_train, y_train)


# y_pred = classifier.predict(x_test)
#cm = confusion_matrix(y_test, y_pred)

# # #apply k-fold cross validation
#accuracies = cross_val_score(regressor, x_train, y_train, cv=10)
# print("Model accuracy: "+"{:.2f}".format(accuracies.mean()) +" ± "+ "{:.2f}".format(accuracies.std()) +" %")


