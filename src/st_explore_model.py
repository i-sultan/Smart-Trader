"""Model file for data exploration
"""

#.-------------------.
#|      imports      |
#'-------------------'
import pandas as pd
from st_helpers import *

#.-----------------.
#|    constants    |
#'-----------------'
moving_average_window = 5

#.----------------.
#|    Classes     |
#'----------------'
class ExploreModel(object):
    """ Model class for explore. Used to explore stocks.

    The model executes requests received from the controller following
    the MVC design pattern.

    Explore model executes the run method of the exploration given by
    the controller.

	"""
    def __init__(self):
        self.explorations_list = [StocksStats, RollingMean, RollingStd, BollingerBands, DailyReturnsPlot, DailyReturnsStats, 
                                  CumulativeReturnsPlot, CumulativeReturnsStats, DailyReturnsHist, CorrelationPlot, CorrelationStats]

class Exploration(object):
	name = ''
	@staticmethod
	def run(df, concat):
		pass

class RollingMean(Exploration):
    name = "Rolling Mean (Moving Average)"

    @staticmethod
    def run(df, concat):
        return_df = df.rolling(window = moving_average_window, center=False).mean()
        return_df.columns = [colname+'-SMA' for colname in return_df.columns]
        if concat:
            return_df = pd.concat([df, return_df], axis = 1)     
        return_df = return_df.reindex_axis(sorted(return_df.columns), axis=1)       
        return_plot = PlotInfo(df = return_df, title = "Simple Moving Average (SMA)", xlabel = "Date", 
							   ylabel = "Price", type = PlotTypes.Plot)
        return return_plot

class RollingStd(Exploration):
    name = "Rolling Standard Deviation"

    @staticmethod
    def run(df, concat):
        return_df = df.rolling(window = moving_average_window, center=False).std()
        return_df.columns = [colname+'-MSTD' for colname in return_df.columns]
        if concat:
			#std should only be concatenated to daily returns and not actual values
            return_df = pd.concat([daily_returns(df), return_df], axis = 1)  
        return_df = return_df.reindex_axis(sorted(return_df.columns), axis=1)  
        return_plot = PlotInfo(df = return_df, title = "Moving Standard Deviation", xlabel = "Date", 
							   ylabel = "Standard Deviation", type = PlotTypes.Plot)
        return return_plot

class BollingerBands(Exploration):
    name = "Bollinger Bands"

    @staticmethod
    def run(df, concat):
        rm = df.rolling(window = moving_average_window, center=False).mean()
        rstd = df.rolling(window = moving_average_window, center=False).std()
        
        return_df_lower = rm - 2 * rstd
        return_df_lower.columns = [colname+'-Lower' for colname in return_df_lower.columns]
        return_df_upper = rm + 2 * rstd
        return_df_upper.columns = [colname+'-Upper' for colname in return_df_upper.columns]
        if concat:
			#concatenate rolling mean in addition to upper and lower bands
            rm.columns = [colname+'-SMA' for colname in rm.columns]
            return_df = pd.concat([df, rm, return_df_lower, return_df_upper], axis = 1)
        else:
            return_df = pd.concat([return_df_lower, return_df_upper], axis = 1)
        return_df = return_df.reindex_axis(sorted(return_df.columns), axis=1) 
        return_plot = PlotInfo(df = return_df, title = "Bollinger Bands", xlabel = "Date", 
							   ylabel = "Price", type = PlotTypes.Plot)
        return return_plot

class DailyReturnsPlot(Exploration):
    name = "Daily Returns Plot"

    @staticmethod
    def run(df, concat):
		#concat is ignored in histogram function
        return_df = daily_percent_returns(df)  
        return_df = return_df.reindex_axis(sorted(return_df.columns), axis=1)          
        return_plot = PlotInfo(df = return_df, title = "Daily Percent Returns Plot", xlabel = "Date", 
							   ylabel = "Daily percent return(%)", type = PlotTypes.Plot)
        return return_plot

class CumulativeReturnsPlot(Exploration):
    name = "Cumulative Returns Plot"

    @staticmethod
    def run(df, concat):
		#concat is ignored in histogram function
        return_df = cumulative_percent_returns(df)  
        return_df = return_df.reindex_axis(sorted(return_df.columns), axis=1)          
        return_plot = PlotInfo(df = return_df, title = "Cumulative Percent Returns Plot", 
                               xlabel = "Cummulative percent return(%)", ylabel = "Price", type = PlotTypes.Plot)
        return return_plot

class DailyReturnsHist(Exploration):
    name = "Daily Returns Histogram"

    @staticmethod
    def run(df, concat):
		#concat is ignored in histogram function
        return_df = daily_percent_returns(df)
        return_df = return_df.reindex_axis(sorted(return_df.columns), axis=1)          
        return_plot = PlotInfo(df = return_df, title = "Daily Percent Returns Histogram", 
                               xlabel = "Daily percent return(%)", ylabel = "Count", type = PlotTypes.Hist)
        return return_plot

class CorrelationPlot(Exploration):
    name = "Scatter Plot"

    @staticmethod
    def run(df, concat):
		#we only plot correlation of the first two stocks in the list. TODO: check user experience
        return_df = daily_percent_returns(df.ix[:,[0,1]])
        return_df = return_df.reindex_axis(sorted(return_df.columns), axis=1)          
        return_plot = PlotInfo(df = return_df, title = "Scatter Plot of {} vs {} Daily Percent Returns".format(return_df.columns[1],return_df.columns[0]), 
							   xlabel = return_df.columns[0] + " daily percent return(%)", ylabel = return_df.columns[1] + " daily percent return(%)", 
                               type = PlotTypes.Scatter)
        return return_plot

class StocksStats(Exploration):
    name = "Stock Stats"

    @staticmethod
    def run(df, concat):
        return_table = TableInfo(title = "Stock Stats", df = df.describe().append(calc_sharpe(df)))
        return return_table

class DailyReturnsStats(Exploration):
    name = "Daily Returns Stats"

    @staticmethod
    def run(df, concat):
        return_df = daily_percent_returns(df)  
        return_table = TableInfo(title = "Daily Return Stats", df = return_df.describe().append(calc_sharpe(df)))
        return return_table

class CumulativeReturnsStats(Exploration):
    name = "Cumulative Returns Stats"

    @staticmethod
    def run(df, concat):
        return_df = cumulative_percent_returns(df)  
        return_table = TableInfo(title = "Cumulative Return Stats", df = return_df.describe())
        return return_table

class CorrelationStats(Exploration):
    name = "Correlation Stats"

    @staticmethod
    def run(df, concat):
        return_table = TableInfo(title = "Correlation stats", df = df.corr())
        return return_table