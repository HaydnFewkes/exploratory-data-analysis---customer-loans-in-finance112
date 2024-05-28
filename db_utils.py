import yaml
from sqlalchemy import create_engine, inspect
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PowerTransformer
import seaborn as sns

# Opens the database credentails, and saves them
def GetCreds():
    with open('credentials.yaml', 'r') as file:
        creds = yaml.safe_load(file)
        return creds


class RDSDatabaseConnector:
    """
    Class for connecting to said database
    """
    def __init__(self, creds):
        self.creds = creds

    def RDSConnection(self):
        """
        Establishes the connection
        """
        self.DATABASE_TYPE = 'postgresql'
        self.DBAPI = 'psycopg2'
        self.HOST = self.creds['RDS_HOST']
        self.USER = self.creds['RDS_USER']
        self.PASSWORD = self.creds['RDS_PASSWORD']
        self.DATABASE = self.creds['RDS_DATABASE']
        self.PORT = self.creds['RDS_PORT']
        self.engine = create_engine(f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}")

    def RDSExtract(self):
        """
        Extracts the data from the db to a pandas frame
        """
        connection = self.engine.connect()
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()
        self.loan_payments = pd.read_sql_table(f"{table_names[0]}", self.engine)
        connection.close()

    def RDSSaveToCSV(self):
        """
        Saves the dataframe as a csv locally for ease of access
        """
        self.loan_payments.to_csv('loan_payments.csv', index=False)


class DataTransform:
    """
    Transforms alot of the data into an easier or better suited format
    """
    def __init__(self, database):
        self.df = database

    def convert_to_datetime(self, column_names):
        """
        Converts all string dates into datetime format
        """
        for column in column_names:
            self.df[column] = pd.to_datetime(self.df[column])

    def convert_to_int(self):
        """
        Converts any string columns that better fit the integer datatype into said datatype
        """
        self.df['term'] = self.df['term'].str.replace('months','')
        self.df['term'] = self.df['term'].astype(float)
        self.df.rename(columns={'term':'term_in_mnths'})

class DataFrameInfo:
    """
    Returns alot of info about the dataframe
    """
    def __init__(self, database):
        self.df = database

    def check_datatypes(self):
        """
        Returns datatypes for the columns
        """
        return self.df.dtypes

    def get_mean(self):
        """
        Returns the mean for all numeric columns
        """
        return self.df.mean(skipna=True, numeric_only=True)
    
    def get_median(self):
        """
        Returns the median for all numeric columns
        """
        return self.df.median(axis=0, numeric_only=True)
    
    def get_standard_deviation(self):
        """"
        Returns the stand deviation for all numeric columns
        """
        return self.df.std(skipna=True,numeric_only=True)
    
    def get_distinct_values(self):
        """
        Returns the amount of unique values in each column
        """
        return self.df.nunique()

    def get_shape(self):
        """
        Returns the shape (size) of the dataframe
        """
        return self.df.shape

    def get_null_values(self):
        """
        Returns the count od null values in each column
        """
        return ((self.df.isna().sum()/self.df.shape[0])*100)
    
class DataFrameTransform:
    """
    Used for more drastic changes to the data
    """
    def __init__(self, df):
        self.df = df
        self.null_percent = ((self.df.isna().sum()/self.df.shape[0])*100)
        self.skew = self.df.skew(numeric_only=True)
        self.numerical_cols = 0
        self.newdf = self.df

    def null_values(self):
        """
        Removes all columns with higher than 50% null value count
        """
        count = -1
        for column in self.null_percent:
            count += 1
            if column >= 50:
                self.df = self.df.drop(self.df.columns[count], axis = 1)
                count -= 1
    
    def data_impute(self, column_names, data_type):
        """
        Replaces all null values with either the mean or mode,
        which ever is more suited
        """
        if data_type == 'mean':
            for column in column_names:
                self.df[column] = self.df[column].fillna(self.df[column].mean(skipna=True))
            self.null_percent = ((self.df.isna().sum()/self.df.shape[0])*100)
        if data_type == 'mode':
            for column in column_names:
                self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
            self.null_percent = ((self.df.isna().sum()/self.df.shape[0])*100)
        
    def correct_skew(self):
        """
        Corrects the skew of the dataframe
        """
        self.numerical_cols = self.df.select_dtypes(include=['number'])
        self.skew = self.df.skew(numeric_only=True)
        pt = PowerTransformer()
        for column in self.numerical_cols:
            if self.skew[column] >= 1:
                temp_df = self.numerical_cols[column].to_frame()
                pt.fit(temp_df)
                self.newdf[column] = pt.transform(temp_df)
            
    def correlation(self):
        """
        Removes the overly correlated columns
        In this case we only has one column that was an issue
        """
        self.newdf = self.newdf.drop(columns=['delinq_2yrs'])
        print(self.newdf.columns.tolist())

class Plotter:
    """
    Class for visulising the data
    """
    def __init__(self, df):
        self.df = df

    def plot_frame(self):
        """
        Plots the frame of the data
        """
        msno.matrix(self.df)
        plt.show()

    def skewness(self):
        """
        Plots the skewness
        """
        self.skew = self.df.skew(numeric_only=True)
        #print(self.skew)
        plt.plot(self.skew)
        plt.show()

    def corr_matrix(self):
        """
        Plots a correlation matrix
        """
        sns.set_theme(style='white')
        corr = self.df.corr(numeric_only = True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()

creds = GetCreds()
a = RDSDatabaseConnector(creds)
a.RDSConnection()
a.RDSExtract()
#a.RDSSaveToCSV()

b = DataTransform(a.loan_payments)
b.convert_to_datetime(['issue_date','last_payment_date','next_payment_date','earliest_credit_line'])
b.convert_to_int()

#c = DataFrameInfo(b.df)
#print(c.get_null_values())
#print(c.get_shape())

d = DataFrameTransform(b.df)
d.null_values()
e = Plotter(d.df)
#e.plot_frame()
d.data_impute(['funded_amount','term', 'int_rate','collections_12_mths_ex_med'], 'mean')
#e.plot_frame()
d.data_impute(['employment_length', 'last_payment_date', 'last_credit_pull_date'], 'mode')
#e.skewness()
d.correct_skew()
f = Plotter(d.newdf)
d.correlation()
#f.skewness()
#f.corr_matrix()
#e.plot_frame()
#print(d.null_percent)
#USE IPYNB!!!!!