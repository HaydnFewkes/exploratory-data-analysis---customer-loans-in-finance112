from sklearn.preprocessing import PowerTransformer

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