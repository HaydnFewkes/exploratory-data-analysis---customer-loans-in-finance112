import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np

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