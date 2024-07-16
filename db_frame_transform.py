from sklearn.preprocessing import PowerTransformer
import numpy as np
from scipy import stats

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
        #Recommended values to use:
        # (['funded_amount','term', 'int_rate','collections_12_mths_ex_med'], 'mean')
        # (['employment_length', 'last_payment_date', 'last_credit_pull_date'], 'mode')
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
        self.numerical_cols = self.df[['loan_amount','funded_amount','funded_amount_inv','int_rate','instalment','dti','open_accounts','total_accounts','out_prncp','out_prncp_inv','total_payment','total_payment_inv','total_rec_prncp','total_rec_int','last_payment_amount']]
        pt = PowerTransformer()
        for column in self.numerical_cols.columns:
            temp_df = self.numerical_cols[column].to_frame()
            pt.fit(temp_df)
            self.newdf[column] = pt.transform(temp_df)
            
    def correlation(self):
        """
        Removes the overly correlated columns
        """
        self.newdf = self.newdf.drop(columns=['member_id','total_payment_inv','funded_amount_inv','funded_amount'
                                              ,'out_prncp_inv','instalment','recoveries','total_rec_prncp'])
        
    def outliers(self):
        self.numerical_cols = self.df[['loan_amount','funded_amount','funded_amount_inv','int_rate','instalment','dti','open_accounts','total_accounts','out_prncp','out_prncp_inv','total_payment','total_payment_inv','total_rec_prncp','total_rec_int','last_payment_amount']]
        #for column in self.numerical_cols.columns:
        #    Q1 = np.quantile(self.df[column], 0.25)
        #    Q3 = np.quantile(self.df[column], 0.75)
        #    IQR = Q3-Q1
        #    counter = 0
        #    for value in self.numerical_cols[column]:
        #        if value > Q3+IQR*1.5 or value < Q1-IQR*1.5:
        #            self.df.drop([counter])
        #        counter += 1
        self.df = self.df[(np.abs(stats.zscore(self.numerical_cols)) < 3).all(axis=1)]
        print(self.df.shape)
            