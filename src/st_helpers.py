""" File for helper functions used by more than one class in ST system.
"""

#.-------------------.
#|      imports      |
#'-------------------'

import pandas as pd
import numpy as np
import pandas_datareader.data as web 
import matplotlib.pyplot as plt
from cycler import cycler
from pandas.tools.plotting import table

#.---------------------.
#|  exception classes  |
#'---------------------'

class Error(Exception):
    """Base class for exceptions."""
    pass

class ValidationError(Error):
    """Exception raised for errors in the input.

    Attributes:
        - message (str): explanation of the error.

    """
    def __init__(self, message):
        self.message = message

#.---------------.
#|    classes    |
#'---------------'
class PlotTypes:
    Plot, Hist, Scatter, Table = range(4)

class PlotInfo(object):
    """Class that contains all info needed for plotting"""
    def __init__(self, df, title, xlabel, ylabel, type, line_styles = None):
        self.df = df
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.type = type
        self.line_styles = line_styles

class TableInfo(object):
    """Class that contains all info needed for table printing"""
    def __init__(self, df, title):
        self.df = df
        self.title = title
        self.type = PlotTypes.Table

class StColors(object):
    orange       = '#f5be2e'
    bright_green = '#b7f731'
    dark_grey    = '#191919'
    mid_grey     = '#323232'
    light_grey   = '#c8c8c8'

#.-------------------------.
#|    helper functions     |
#'-------------------------'

def get_data_web(symbols, dates, force_spy = True) :
    """Read stock data (adjusted close) for given symbols from web """
    df = pd.DataFrame(index=dates)

    if force_spy and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')
    
    for symbol in symbols:
        append_days = 21 #add enough days to cover first na's of the moving window
        df_temp = web.DataReader(symbol, data_source='yahoo', start=dates[0]-append_days, end=dates[-1])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        if len(symbols) == 1:
            #indicators: Daily, SMA(%), EMA(%), SMA Momentum, Volatility, Volume
            df = df_temp[[symbol, 'Volume', 'Open', 'Close', 'Low', 'High']]
            pd.options.mode.chained_assignment = None  #removes unnecessary warning
            df['Volume'] = df['Volume'].pct_change(periods=1)
            df['Open'] = (df['Open']-df['Close'])/df['Close']
            df['High'] = (df['High']-df['Close'])/df['Close']
            df['Low']  = (df['Low' ]-df['Close'])/df['Close']
            df['SMA'] = df[symbol].pct_change(periods=1).rolling(window = 10).mean()
            df['EWMA']= df[symbol].pct_change(periods=1).ewm(span = 10).mean()
            df['MOM'] = df[symbol].pct_change(periods=10).rolling(window = 10).mean()
            df['STD'] = df[symbol].pct_change(periods=1).rolling(window = 10).std()
        else:
            df = df.join(df_temp[[symbol]])
        df = df.ix[append_days:,:]
        if force_spy and symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=['SPY'])
    if not force_spy:
        df = df.dropna(subset=[df.columns[0]])
    return df

def get_tableau_colors():
    tableau20 = [(31, 119, 180) , (174, 199, 232), (255, 127, 14) , (255, 187, 120), (44 , 160, 44 ), 
				 (152, 223, 138), (214, 39, 40)  , (255, 152, 150), (148, 103, 189), (197, 176, 213), 
				 (140, 86, 75)  , (196, 156, 148), (227, 119, 194), (247, 182, 210), (127, 127, 127), 
				 (199, 199, 199), (188, 189, 34) , (219, 219, 141), (23, 190, 207) , (158, 218, 229)]    
    return [(r/255., g/255., b/255.) for (r,g,b) in tableau20]
    
def plot_data(fig, plot_info):
    """Plot with specified title and axis labels."""
    ax = fig.add_subplot(111)

	#configure surrounding area (edges, ticks, colors)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)  
    ax.set_prop_cycle(cycler('color', get_tableau_colors()))
    
    #plot data
    df = plot_info.df
    if plot_info.type == PlotTypes.Plot:
        if plot_info.line_styles:
            df.plot(ax=ax, fontsize=12, lw=2.5, style = plot_info.line_styles)
        else:
            df.plot(ax=ax, fontsize=12, lw=2.5)
    elif plot_info.type == PlotTypes.Hist:
        df.plot.hist(ax=ax, fontsize=12, bins = 25, alpha = .8, lw=1)
    elif plot_info.type == PlotTypes.Scatter:
        df.plot.scatter(x = df.columns[0], y = df.columns[1], ax=ax, fontsize=12, lw=.5)
		#plot linear fit on top of scatter plot, and display alpha\beta
        beta, alpha = np.polyfit(df.ix[:,0], df.ix[:,1], 1)
        ax.plot(df.ix[:,0], beta * df.ix[:,0] + alpha, '-', lw = 3, c = (1, 0.5, 0.17))
        text = r'$\alpha = $' + '%.2f\n'%alpha + r'$\beta = $' + '%.2f'%beta
        props = dict(boxstyle='round', facecolor=(1,0.5,0.17), alpha=.7)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

	#configure interior (legend, grid)
    if plot_info.type != PlotTypes.Scatter:
        ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    ax.xaxis.grid(True,  linestyle='--', alpha =.1, lw=1)
    ax.yaxis.grid(True,  linestyle='--', alpha =.2, lw=1)

    #set title and x,y labels
    fig.suptitle(plot_info.title, fontsize=20, fontweight='bold')
    ax.set_xlabel(plot_info.xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel(plot_info.ylabel, fontsize=16, fontweight='bold')    

	#setup padding
    fig.tight_layout()
    fig.subplots_adjust(top=0.9) #set axes area percentage of the figure

def tabulate_data(fig, table_info):
    """Prepare a matplotlib table using provided table info and adding result to figure."""
    fig.suptitle(table_info.title, fontsize=20, fontweight='bold')
    ax = fig.add_subplot(111)
   
	#configure table colors
    tableau20 = get_tableau_colors()
    color_1 = tableau20[0]
    color_2 = tableau20[1]

	#setup table at the middle of the figure
    df = table_info.df
    df.index = ' ' + df.index + '    ' #adding spaces to index(label) column since label column is fixed width 
    nrows, ncols = df.shape
    colwidth = 0.16
    rowheight = 0.1
    tab = table(ax, np.round(df, 2), loc='upper center', bbox=[.5-ncols*colwidth/2,.5-nrows*rowheight/2,ncols*colwidth,nrows*rowheight])

    for key, cell in tab.get_celld().items():
	    #set cell properties
        cell._text.set_size(14)
        cell.set_edgecolor('w')
        cell.set_linestyle('-')
        cell.set_facecolor('w')
        cell.set_linewidth(1)
        #change color of even rows vs. odd rows 
        row, col = key
        if row%2 == 0:
            cell.set_facecolor(color_1)
            cell._text.set_color('w')
        else:
            cell.set_facecolor(color_2)
            cell._text.set_color([i*0.65 for i in color_1])
	    #set color for header and index column
        if row == 0 or col == -1:
            cell._text.set_color('w')
            cell._text.set_weight('bold')
            cell.set_facecolor([i*0.65 for i in color_1])
        if row == 0:
            cell.set_height(cell.get_height()*1.4) #makes first row a bit taller

    ax.axis('off')

def daily_percent_returns(df):
    daily_percent_returns = df.pct_change(periods=1)*100
    daily_percent_returns.ix[0,:] = 0
    return daily_percent_returns

def daily_returns(df):
    daily_returns = df-df.shift(1) 
    daily_returns.ix[0,:] = 0
    return daily_returns

def cumulative_percent_returns(df):
    cumulative_returns = (df/df.ix[0]-1)*100
    return cumulative_returns

def calc_sharpe(df, risk_free_rate = 0): #annual risk_free_rate of 5% should be given as 0.05
    df = df.pct_change() - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * df.mean() / df.std()
    sharpe_ratio.name = 'Sharpe ratio'
    return sharpe_ratio

def calc_errors(error_list, relative_error_list, hit_list):
    """ Calculate different types of errors (MSE, RMSE, MAE, MAPE, hit_rate) 
        based on lists of error and relative errors
    """
    absolute_error_list = np.abs(error_list)
    MSE = np.mean(absolute_error_list**2)
    RMSE= np.sqrt(MSE)
    MAE = np.mean(absolute_error_list)
    MAPE= np.mean(np.abs(relative_error_list))
    hit_rate = float(sum(hit_list))/len(hit_list)
    return (MSE, RMSE, MAE, MAPE, hit_rate)