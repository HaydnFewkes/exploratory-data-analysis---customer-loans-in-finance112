import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

class Plotter:
    """
    Class for visulising the data
    """
    def __init__(self, df):
        self.df = df

    def reset_frame(self, df):
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
        self.numerical_cols = self.df[['loan_amount','funded_amount','funded_amount_inv','int_rate','instalment','dti','open_accounts','total_accounts','out_prncp','out_prncp_inv','total_payment','total_payment_inv','total_rec_prncp','total_rec_int','last_payment_amount']]
        for column in self.numerical_cols.columns:
            plt.title(str(column))
            plt.hist(self.df[column])
            plt.show()
        #self.numerical_cols = self.df.select_dtypes(include=['number'])
        #for column in self.numerical_cols:
        #    self.skew = column.skew()
        #    print(self.skew)
        

    def corr_matrix(self):
        """
        Plots a correlation matrix
        """
        corr = self.df.corr(numeric_only=True)
        plt.subplots(figsize=(20,15))
        sns.heatmap(corr, annot=True, linewidths=0.5)
        plt.show()