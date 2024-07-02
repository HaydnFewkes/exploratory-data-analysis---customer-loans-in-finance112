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
        self.skew = self.df.skew(numeric_only=True)
        plt.plot(self.skew)
        plt.show()

    def corr_matrix(self):
        """
        Plots a correlation matrix
        """
        corr = self.df.corr(numeric_only=True)
        plt.subplots(figsize=(20,15))
        sns.heatmap(corr, annot=True, linewidths=0.5)
        plt.show()