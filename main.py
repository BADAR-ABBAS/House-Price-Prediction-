import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler

df = pd.read_csv('House Price Prediction Dataset.csv')



def Feature_Engineering(data_in_df):

    encoder = OneHotEncoder(sparse_output=False , drop='first').set_output(transform="pandas")
    encoded_data = encoder.fit_transform(data_in_df)
    return encoded_data


def Feature_Scaling(variable_data , numerical_data):
    
    engineered_data = Feature_Engineering(variable_data)
    scaling_data = pd.concat([numerical_data,engineered_data],axis=1)
    print(scaling_data.columns.tolist())
    std_scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        std_scaler.fit_transform(scaling_data),
        columns=scaling_data.columns
    )

    return df_standardized


def Regression(x,y):

    x_train , x_test , y_train , y_test = train_test_split (x,y,random_state=42)

    model = LinearRegression()

    model.fit(x_train,y_train)
    pred = model.predict(x_test)

    mse = mean_squared_error(y_test,pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test,pred)

    print (f"Mean square error : {mse}")
    print (f"root Mean square error : {rmse}")
    print (f"R2 score {r2}")

variable_data = df [['Location','Condition','Garage']]
numerical_data = df [['Area','Bedrooms','Bathrooms','Floors','YearBuilt']]
fs = Feature_Scaling(variable_data , numerical_data)

x = fs
y = df[['Price']]
Regression(x,y)

# Print linear correlation values
# print(df.corr(numeric_only=True)['Price'].sort_values(ascending=False))










def Showdata():
    #  Id  Area  Bedrooms  Bathrooms  Floors  YearBuilt  Location  Condition Garage   Price
    # print(df.shape)
    # print(df.isnull().sum())
    # print(df.dtypes)
    # print(df.head())
    # print(df.info())
    # plt.figure(figsize=(8,5))
    x = fs[['Area' ]]
    y = fs[['Price']]

    fig , ax = plt.subplots()
    ax.plot(x,y,'x')
    ax.set_title("House price prediction")
    ax.set_xlabel("Area")
    ax.set_ylabel("Price")
    # ax.set_xticks([1,2,3,4,5])
    plt.show()

    # sns.boxplot(x='Floors',y='Price',data=df)
    # plt.xlabel("Bedrooms")
    # plt.ylabel("Price")
    # plt.show()


# Showdata()