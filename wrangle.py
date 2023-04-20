# standard imports
import pandas as pd
# visualization
import matplotlib.pyplot as plt
import seaborn as sns
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
# credentials
from env import user, password, host


def acquire_zillow():
    ''' Acquires data from Zillow using env imports and renames columns'''
    url = f'mysql+pymysql://{user}:{password}@{host}/zillow'
    query = '''
SELECT bedroomcnt
, bathroomcnt
, calculatedfinishedsquarefeet
, taxvaluedollarcnt
, yearbuilt
, taxamount
, fips
, regionidzip
FROM properties_2017
LEFT JOIN propertylandusetype USING(propertylandusetypeid)
WHERE propertylandusedesc IN ("Single Family Residential"
                                , "Inferred Single Family Residential")'''
    # return dataframe of zillow data
    df = pd.read_sql(query, url)
    # rename columns
    df = df.rename(columns= {'bedroomcnt': 'bedrooms'
                         , 'bathroomcnt': 'bathrooms'
                         , 'calculatedfinishedsquarefeet': 'sqr_ft'
                         , 'taxvaluedollarcnt': 'tax_value'
                         , 'yearbuilt': 'year_built'
                         , 'taxamount': 'tax_amount'
                         , 'regionidzip': 'zipcode'})
    # drop null values in zipcode
    df.zipcode = df.zipcode.astype(object).dropna(axis=0)
    return df

def remove_outliers(df, k, col_list):
    ''' removes outliers from a list of columns in a dataframe
        and returns that dataframe'''
    for col in col_list:
        # get quartiles
        q1, q3 = df[col].quantile([.25, .75])
        # calculate interquartile range
        iqr = q3 - q1
        # get upper bound
        upper_bound = q3 + k * iqr
        # get lower bound
        lower_bound = q1 - k * iqr
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df

def get_hist(df):
    ''' Returns histographs of acquires continuous variables'''
    plt.figure(figsize=(16, 3))
    # list of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built', 'zipcode']]
    for i, col in enumerate(cols):
        # i starts at 0, but plots nos should start at 1
        plot_number = i + 1
        # create subplot
        plt.subplot(1, len(cols), plot_number)
        # title with column name
        plt.title(col)
        # display histogram for column
        df[col].hist(bins=5)
        # hide gridelines
        plt.grid(False)
        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)
        # set proper spacing between plots
        plt.tight_layout()
    plt.show()
    
def get_box(df):
    ''' Returns boxplots of acquire continuous variables'''
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'sqr_ft', 'tax_value', 'tax_amount']
    # figure size
    plt.figure(figsize=(16, 3))
    for i, col in enumerate(cols):
        # i starts at 0, but plots nos should start at 1
        plot_number = i + 1
        # create subplot
        plt.subplot(1, len(cols), plot_number)
        # title with column name
        plt.title(col)
        # display boxplot for column
        sns.boxplot(data=df[[col]])
        # hide gridelines
        plt.grid(False)
        # set proper spacing between plots
        plt.tight_layout()
    plt.show()
    
def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''
    # remove outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'sqr_ft', 'tax_value', 'tax_amount'])
    # get distributions of numeric data
    get_hist(df)
    get_box(df)
    # converting column datatypes
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    #train, validate, test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train[['year_built']])
    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])
    return train, validate, test

def wrangle_zillow():
    ''' Acquire and prepare data from Zillow batabase for explore'''
    train, validate, test = prepare_zillow(acquire_zillow())
    # inital glace at data
    print('_'*50)
    print(f'Data Frame: \n{train.sort_index().head(2).T.to_markdown()}')
    print('_'*50)
    print(f'Stats: \n{train.describe().T}')
    print('_'*50)
    print('Info: ')
    print(train.info())
    print('_'*50)
    print(f'Data Types: \n{train.dtypes}')
    print('_'*50)
    print(f'Null Values: \n{train.isnull().sum()}')
    print('_'*50)
    print(f'NA Values: \n{train.isna().sum()}')
    print('_'*50)
    print(f'Unique Value Count: \n{train.nunique()}')
    return train, validate, test

