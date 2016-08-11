"""Controller file for backtest model and view
"""

#.-------------------.
#|      imports      |
#'-------------------'
from st_train_model import *
from st_backtest_view import *
from st_cache_handler import *
from st_helpers import *

#.----------------.
#|    Classes     |
#'----------------'
class BacktestController(object):
    """ Controller class for backtest model and view.

    The controller handles communications between model and view objects following
    the MVC design pattern.

    Args:
        - parent_view (tk widget): the view object which the backtest view will
        be part of.

	"""

    def __init__(self, parent_view):
        self.cache = CacheHandler()
        self.model = ModelsCollection()        
        self.view  = BacktestView(parent_view, self)

    def get_trained_symbols(self):
        return self.cache.get_folders()

    def get_trained_models(self, symbol):
        return self.cache.get_subfolders(symbol)

    def run_backtest(self, symbol, model, start_date, end_date):
        """ Test model on a stock symbol from start_date to end_date, generate a plot of
        testing results, and save the training summary.
        
        Args:
            - symbol (str): stock symbol to build the model for
            - model (PredictiveModel): predictive model to be used in testing
            - start_date (Date): first date of training period.
            - end_date (Date): last date of training period
        
        Returns:
            - training_summary (TableInfo): containing model, tested params, and testing metrics
        """
        if symbol not in self.cache.get_folders():
            raise ValidationError('Symbol {} not found'.format(symbol))
        # add 1 year ahead of starting date for training
        dates = pd.date_range(start_date, end_date).union(pd.date_range(end = start_date, periods=1855, freq='D')) #~5 years of bootstrapping
        df = get_data_web([symbol], dates, False)
        training_summary = self.cache.load_single(symbol, model)
        if training_summary.empty: #file failed to load
            raise ValidationError('No training summary file found for the specified symbol and model.')
        num_models = 1

        plot, metrics_df = self.model.backtest(df, training_summary, num_models)
        
        folder_name = symbol
        subfolder_name = model
        file_name = start_date + "__" + end_date + "_test_metrics.csv"
        self.cache.save_df(folder_name, subfolder_name, file_name, metrics_df)
        
        return plot