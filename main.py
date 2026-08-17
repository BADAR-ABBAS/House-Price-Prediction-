import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

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

    param_grid = {
        'max_depth': [ 3, 4, 5, 6, 7, 8, 9, 10 , None],
        'min_samples_split': [2, 3, 4, 5,6,7,8,9,10],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': [100, 200]
    }

    #Find Out parameters for max_depth and min_samples_split for Random Forest Regressor
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        )
    
    grid_search.fit(x_train, y_train.values.ravel())
    RDF_best_model = grid_search.best_estimator_
    print("Best parameters found: ", grid_search.best_params_)



    # Find Parameters for max_depth and min_samples_split for Gradient Boosting Regressor
    param_grid_gb = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 4, 6, 8],
    'subsample': [0.7, 0.8, 1.0]
    }

    grid_search_gb = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid=param_grid_gb,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    grid_search_gb.fit(x_train, y_train.values.ravel())

    print("Best GB parameters found:", grid_search_gb.best_params_)
    GB_best_model = grid_search_gb.best_estimator_

    #--------------------------------------------------------------------
    model = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=RDF_best_model.n_estimators,
                                                random_state=42,
                                                max_depth=RDF_best_model.max_depth,
                                                min_samples_split=RDF_best_model.min_samples_split
                                                ),
        "Gradient Boosting": GradientBoostingRegressor(learning_rate=GB_best_model.learning_rate, 
                                                       max_depth=GB_best_model.max_depth, 
                                                       min_samples_split=GB_best_model.min_samples_split, 
                                                       n_estimators=GB_best_model.n_estimators, 
                                                       subsample=GB_best_model.subsample, 
                                                       random_state=42
                                                       )
    }

    for name, model in model.items():
        model.fit(x_train,y_train.values.ravel())
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
    # print(df.head())
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
    # plt.show()

    # sns.boxplot(x='Floors',y='Price',data=df)
    # plt.xlabel("Bedrooms")
    # plt.ylabel("Price")
    # plt.show()


Showdata()