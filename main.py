import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

df = pd.read_csv('house_prices_prediction.csv')



def Feature_Engineering(data_in_df):

    encoder = OneHotEncoder(sparse_output=False , drop='first').set_output(transform="pandas")
    encoded_data = encoder.fit_transform(data_in_df)
    return encoded_data

def Feature_Scaling(variable_data , numerical_data):
    
    engineered_data = Feature_Engineering(variable_data)
    scaling_data = pd.concat([numerical_data,engineered_data],axis=1)
    std_scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        std_scaler.fit_transform(scaling_data),
        columns=scaling_data.columns
    )

    return df_standardized

def plot_pred_vs_actual(y_test,pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs. Predicted House Prices')
    plt.grid(True)
    plt.show()


def Regression(x,y):
    x_train , x_test , y_train , y_test = train_test_split (x,y,random_state=42)
    model = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    for name, model in model.items():
        model.fit(x_train,y_train)
        pred = model.predict(x_test)

        mse = mean_squared_error(y_test,pred)
        rmse = mse ** 0.5
        test_r2 = r2_score(y_test,pred)
        train_score = model.score(x_train,y_train)

        print (f"Mean square error : {name} : {mse}")
        print (f"root Mean square error : {name} : {rmse}")
        print (f"Test R2 score {name} : {test_r2}")
        print (f"Train R2 score {name} : {train_score}")
        # plot_pred_vs_actual(y_test, pred)


variable_data = df [[]]
numerical_data = df [['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','YearBuilt','FullBath']]
fs = Feature_Scaling(variable_data , numerical_data)

x = fs
y = df[['SalePrice']]
Regression(x,y)


# Print linear correlation values
# print(df.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False))










def Showdata():
    #  Id  Area  Bedrooms  Bathrooms  Floors  YearBuilt  Location  Condition Garage   Price
    print(df.shape)
    # print(df.isnull().sum())
    # print(df.dtypes)
    print(df.head())
    # print(df.info())
    # plt.figure(figsize=(8,5))
    # x = fs[['LotArea' ]]
    # y = fs[['SalePrice']]

    # fig , ax = plt.subplots()
    # ax.plot(x,y,'x')
    # ax.set_title("House price prediction")
    # ax.set_xlabel("Area")
    # ax.set_ylabel("Price")
    # ax.set_xticks([1,2,3,4,5])
    plt.show()

    # sns.boxplot(x='Floors',y='Price',data=df)
    # plt.xlabel("Bedrooms")
    # plt.ylabel("Price")
    # plt.show()


# Showdata()