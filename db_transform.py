import pandas as pd


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
        # Recommned values:
        # ['issue_date','last_payment_date','next_payment_date','earliest_credit_line']
        for column in column_names:
            self.df[column] = pd.to_datetime(self.df[column],format="mixed", dayfirst=True)

    def convert_to_int(self):
        """
        Converts any string columns that better fit the float datatype into said datatype
        """
        self.df['term'] = self.df['term'].str.replace('months','')
        self.df['term'] = self.df['term'].astype(float)

    def convert_int_to_flt(self):
        """
        Converts all integers into floats, due to floats being easier to use in calculations
        """
        for column in self.df.select_dtypes(int):
            self.df[column] = self.df[column].astype('float')