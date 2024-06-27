

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