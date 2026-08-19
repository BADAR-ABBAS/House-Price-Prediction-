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
from sklearn.linear_model import Ridge, Lasso



def Feature_Engineering(x_train_variable_data , x_train_numerical_data , x_test_variable_data , x_test_numerical_data):

    encoder = OneHotEncoder(sparse_output=False , drop='first' , handle_unknown='ignore').set_output(transform="pandas")
    encoded_data = encoder.fit_transform(x_train_variable_data)
    X_train = pd.concat([x_train_numerical_data,encoded_data],axis=1)
    X_test = pd.concat([x_test_numerical_data,encoder.transform(x_test_variable_data)],axis=1)
    return X_train, X_test

def Feature_Scaling(X_train , X_test):
    
    std_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        std_scaler.fit_transform(X_train),
        columns=X_train.columns , 
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
            std_scaler.transform(X_test),
            columns=X_test.columns , 
            index=X_test.index
        )

    return X_train_scaled, X_test_scaled

def plot_pred_vs_actual(y_test,pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs. Predicted House Prices')
    plt.grid(True)
    plt.show()


def Regression(y_train, y_test, x_train_variable_data, x_train_numerical_data, x_test_variable_data, x_test_numerical_data):
    x_train , x_test = Feature_Engineering(x_train_variable_data , x_train_numerical_data , x_test_variable_data , x_test_numerical_data)
    

    scaled_x_train , scaled_x_test = Feature_Scaling(x_train , x_test)

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
    
    grid_search.fit(scaled_x_train, y_train.values.ravel())
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

    grid_search_gb.fit(scaled_x_train, y_train.values.ravel())

    print("Best GB parameters found:", grid_search_gb.best_params_)
    GB_best_model = grid_search_gb.best_estimator_

    #--------------------------------------------------------------------
    model = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42),
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
        model.fit(scaled_x_train,y_train.values.ravel())
        pred = model.predict(scaled_x_test)

        mse = mean_squared_error(y_test,pred)
        rmse = mse ** 0.5
        test_r2 = r2_score(y_test,pred)
        train_score = model.score(scaled_x_train,y_train)

        print (f"Mean square error : {name} : {mse}")
        print (f"root Mean square error : {name} : {rmse}")
        print (f"Test R2 score {name} : {test_r2}")
        print (f"Train R2 score {name} : {train_score}")
        # plot_pred_vs_actual(y_test, pred)

df = pd.read_csv('house_prices_prediction.csv')

x = df[['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','YearBuilt','FullBath']]
y = df[['SalePrice']]

x_train , x_test , y_train , y_test = train_test_split (x,y,random_state=42)

x_train_variable_data = x_train[[]]
x_train_numerical_data = x_train[['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','YearBuilt','FullBath']]

x_test_variable_data = x_test[[]]
x_test_numerical_data = x_test[['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','YearBuilt','FullBath']]


Regression(y_train, y_test , x_train_variable_data , x_train_numerical_data , x_test_variable_data , x_test_numerical_data)


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


# Showdata()