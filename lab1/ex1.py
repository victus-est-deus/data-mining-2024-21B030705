import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats

# Ex1: Load the data
df = pd.read_csv('machine-readable-business-employment-data-Jun-2024-quarter.csv')


# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.dtypes)
# print(df.isnull().sum())

# Ex2: Handling missing data

df_dropped = df.dropna()


df_mean_filled = df.fillna({'Data_value': df['Data_value'].mean()})
df_median_filled = df.fillna({'Data_value': df['Data_value'].median()})
df_specific_filled = df.fillna({'Suppressed': 'Not Available'})

df_mean_filled.head()
# df_ffill = df.fillna(method='ffill')
# df_bfill = df.fillna(method='bfill')


# Ex3: Data transformation
scaler = MinMaxScaler()
df['Data_value_normalized'] = scaler.fit_transform(df[['Data_value']])

standardizer = StandardScaler()
df['Data_value_standardized'] = standardizer.fit_transform(df[['Data_value']])

df_encoded = pd.get_dummies(df, columns=['STATUS'])

df['Period_binned'] = pd.cut(df['Period'], bins=[2011, 2013, 2015, 2017, 2019],
                             labels=['2011-2012', '2013-2014', '2015-2016', '2017-2018'])

# Ex4: Feature engineering
df_sorted = df.sort_values(by='Period')


df_sorted['Period_Magnitude_Interaction'] = df_sorted['Period'] * df_sorted['Magnitude']


df_sorted['Growth_rate'] = df_sorted['Data_value'].pct_change(fill_method=None) * 100
df_sorted['Growth_rate'] = df_sorted['Growth_rate'].fillna(0)

#Ex5: Data cleaning
df_cleaned = df.drop_duplicates()


df['z_score'] = stats.zscore(df['Data_value'].dropna())
df_no_outliers = df[(df['z_score'].abs() <= 3) | (df['Data_value'].isnull())]


df['STATUS'] = df['STATUS'].str.upper().str.strip()

# Ex6: Splitting data into training and testing sets

df = df.dropna(subset=['Data_value'])

X = df.drop(['Data_value'], axis=1)
y = df['Data_value']


if y.isnull().sum() > 0:
    print("NaN values detected in the target variable. Please handle missing values.")
else:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print("X_train.shape", X_train.shape)


# Ex7: Building the preprocessing pipeline
    numerical_features = ['Period', 'Magnitude']
    categorical_features = ['STATUS', 'UNITS', 'Subject', 'Group']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])


    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)


    test_score = model_pipeline.score(X_test, y_test)
    print(test_score)
