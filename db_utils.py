import yaml
from sqlalchemy import create_engine, inspect
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

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
    def __init__(self, database):
        self.df = database

    def convert_to_datetime(self, column_names):
        for column in column_names:
            self.df[column] = pd.to_datetime(self.df[column])

    def convert_to_int(self):
        self.df['term'] = self.df['term'].str.replace('months','')
        self.df['term'] = self.df['term'].astype(float)
        self.df.rename(columns={'term':'term_in_mnths'})

class DataFrameInfo:
    def __init__(self, database):
        self.df = database

    def check_datatypes(self):
        return self.df.dtypes

    def get_mean(self):
        return self.df.mean(skipna=True, numeric_only=True)
    
    def get_median(self):
        return self.df.median(axis=0, numeric_only=True)
    
    def get_standard_deviation(self):
        return self.df.std(skipna=True,numeric_only=True)
    
    def get_distinct_values(self):
        return self.df.nunique()

    def get_shape(self):
        return self.df.shape

    def get_null_values(self):
        return ((self.df.isna().sum()/self.df.shape[0])*100)
    
class DataFrameTransform:
    def __init__(self, df):
        self.df = df
        self.null_percent = ((self.df.isna().sum()/self.df.shape[0])*100)

    def null_values(self):
        count = -1
        for column in self.null_percent:
            count += 1
            if column >= 50:
                self.df = self.df.drop(self.df.columns[count], axis = 1)
                count -= 1
    
    def data_impute(self, column_names, data_type):
        if data_type == 'mean':
            for column in column_names:
                self.df[column] = self.df[column].fillna(self.df[column].mean(skipna=True))
            self.null_percent = ((self.df.isna().sum()/self.df.shape[0])*100)
        if data_type == 'mode':
            for column in column_names:
                self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
            self.null_percent = ((self.df.isna().sum()/self.df.shape[0])*100)
        
class Plotter:
    def __init__(self, df):
        self.df = df

    def plot_frame(self):
        msno.matrix(self.df)
        plt.show()

    def skewness(self):
        self.skew = self.df.skew()
        print(self.skew)

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
e.skewness()
#e.plot_frame()
#print(d.null_percent)
#USE IPYNB!!!!!