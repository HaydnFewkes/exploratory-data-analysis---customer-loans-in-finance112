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
        Converts any string columns that better fit the float datatype into said datatype
        """
        self.df['term'] = self.df['term'].str.replace('months','')
        self.df['term'] = self.df['term'].astype(float)

    def convert_int_to_flt(self):
        for column in self.df.select_dtypes(int):
            self.df[column] = self.df[column].astype('float')

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
        
class DataAnalysis:
    def __init__(self, df):
        self.df = df

    def state_of_loans(self):
        loan_percent = (self.df['total_rec_prncp']/self.df['loan_amount'])*100
        sns.relplot(loan_percent)
        plt.show()

    def six_months(self):
        counter = 0
        six_month_percent = []
        for value in self.df['instalment']:
            if ((value*6)+self.df['total_rec_prncp'].iloc[counter]) > self.df['loan_amount'].iloc[counter]:
                six_month_percent.append(self.df['loan_amount'].iloc[counter])
            else:
                six_month_percent.append((value*6)+self.df['total_rec_prncp'].iloc[counter])
            counter += 1
        six_month_percent = pd.Series(six_month_percent)
        loan_percent = (six_month_percent/self.df['loan_amount'])*100
        sns.relplot(loan_percent)
        plt.show()
    
    def loan_loss(self):
        counter = 0
        charged_off_counter = 0
        loan_total = 0
        paid_total = 0
        mnths_paid = 0
        months_left = 0
        total_loss = 0
        for value in self.df['loan_status']:
            if value == 'Charged Off':
                charged_off_counter += 1
                loan_total += self.df['loan_amount'].iloc[counter]
                paid_total += self.df['total_rec_prncp'].iloc[counter]
                mnths_paid = round(self.df['total_payment'].iloc[counter]/self.df['instalment'].iloc[counter])
                mnths_left = self.df['term'].iloc[counter]-mnths_paid
                total_loss += mnths_left*self.df['instalment'].iloc[counter]
            counter += 1
        percent_charged_off = (charged_off_counter/len(self.df['loan_status']))*100
        print('The percent that has been charged off is '+str(round(percent_charged_off,2))+'%'+
              ', and the amount paid back is '+str(round(paid_total,2))+' out of '+str(round(loan_total,2)))
        print('The amount that the company lost due to these charged off loans is, '+str(round(total_loss,2)))

    def possible_loss(self):
        late_count = 0
        counter = 0
        total_loss = 0
        proj_loss = 0
        exp_rev = 0
        for value in self.df['loan_status']:
            if 'Late' in value:
                late_count += 1
                proj_loss += self.df['out_prncp'].iloc[counter]
            elif value == 'Charged Off':
                total_loss += (self.df['loan_amount'].iloc[counter]- self.df['total_payment'].iloc[counter])
            counter += 1
        counter = 0
        for value in self.df['instalment']:
            exp_rev += self.df['term'].iloc[counter]*value
            counter += 1
        late_percent = late_count/len(self.df['loan_status'])*100
        total_loss += proj_loss
        percent_loss = (total_loss/exp_rev)*100
        print('The percentage of users that have a late payment is '+str(round(late_percent,2))+
        '%, the projected loss if said customers were to not finish their payment is '+str(round(proj_loss,2))+
        ' and the percent of loss for all late loans if they were charged off, as well as all currently charged off loans is '
        +str(round(percent_loss,2))+'%')

    def indicator_of_loss(self):
        counter = 0
        grade_count_loss = {
            'A':0,
            'B':0,
            'C':0,
            'D':0,
            'E':0,
            'F':0,
            'G':0
        }
        grade_count = {
            'A':0,
            'B':0,
            'C':0,
            'D':0,
            'E':0,
            'F':0,
            'G':0
        }
        for value in self.df['grade']:
            grade_count[value] += 1
            if self.df['loan_status'].iloc[counter] == 'Charged Off' or 'Late' in self.df['loan_status'].iloc[counter]:
                grade_count_loss[value] += 1
            counter += 1
        grade_percent = {key: round((grade_count_loss[key] / grade_count.get(key, 0))*100,2)
                        for key in grade_count_loss.keys()}
        counter = 0
        purpose_count = {
            'credit_card':0,
            'debt_consolidation':0,
            'home_improvement':0,
            'small_business':0,
            'renewable_energy':0,
            'major_purchase':0,
            'other':0,
            'moving':0,
            'car':0,
            'medical':0,
            'house':0,
            'vacation':0,
            'wedding':0,
            'educational':0
        }
        purpose_count_loss = {
            'credit_card':0,
            'debt_consolidation':0,
            'home_improvement':0,
            'small_business':0,
            'renewable_energy':0,
            'major_purchase':0,
            'other':0,
            'moving':0,
            'car':0,
            'medical':0,
            'house':0,
            'vacation':0,
            'wedding':0,
            'educational':0
        }
        for value in self.df['purpose']:
            purpose_count[value] += 1
            if self.df['loan_status'].iloc[counter] == 'Charged Off' or 'Late' in self.df['loan_status'].iloc[counter]:
                purpose_count_loss[value] += 1
            counter += 1
        purpose_percent = {key: round((purpose_count_loss[key] / purpose_count.get(key, 0))*100,2)
                        for key in purpose_count_loss.keys()}
        counter = 0
        home_count = {
            'MORTGAGE':0,
            'RENT':0,
            'OWN':0,
            'OTHER':0,
            'NONE':0
        }
        home_count_loss = {
            'MORTGAGE':0,
            'RENT':0,
            'OWN':0,
            'OTHER':0,
            'NONE':0
        }
        for value in self.df['home_ownership']:
            home_count[value] += 1
            if self.df['loan_status'].iloc[counter] == 'Charged Off' or 'Late' in self.df['loan_status'].iloc[counter]:
                home_count_loss[value] += 1
            counter += 1
        home_percent = {key: round((home_count_loss[key] / home_count.get(key, 0))*100,2)
                        for key in home_count_loss.keys()}
        print('We can see that the lower the grade level of a loan is more likely to affect if the '+
              'loan is charged off or late ',grade_percent,'.')
        print('The purpose of the loan seems to not have too big of an impact on the likelihood of '+
               'the loan being a loss ',purpose_percent,', execpt if the purpose is small business.')
        print('The home ownership has so imapct on the chance of the loan being '+
               'charged off or late',home_percent)
creds = GetCreds()
a = RDSDatabaseConnector(creds)
a.RDSConnection()
a.RDSExtract()
#a.RDSSaveToCSV()

b = DataTransform(a.loan_payments)
b.convert_to_datetime(['issue_date','last_payment_date','next_payment_date','earliest_credit_line'])
b.convert_to_int()
b.convert_int_to_flt()

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
g = DataAnalysis(d.df)
#d.correlation()
#g.loan_loss()
#g.possible_loss()
g.indicator_of_loss()
#e.state_of_loans()
#f.skewness()
#f.corr_matrix()
#e.plot_frame()
#print(d.null_percent)
#USE IPYNB!!!!!