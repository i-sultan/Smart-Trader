"""Controller file for forecast model and view
"""

#.-------------------.
#|      imports      |
#'-------------------'
from st_train_model import *
from st_forecast_view import *
from st_cache_handler import *
from st_helpers import *

#.----------------.
#|    Classes     |
#'----------------'
class ForecastController(object):
    """ Controller class for forecast model and view 

    The controller handles communications between model and view objects following
    the MVC design pattern.

    Args:
        - parent_view (tk widget): the view object which the backtest view will
        be part of.

	"""

    def __init__(self, parent_view):
        self.cache = CacheHandler()
        self.model = ModelsCollection()        
        self.view  = ForecastView(parent_view, self)        

    def get_trained_symbols(self):
        return self.cache.get_folders()

    def get_trained_models(self, symbol):
        return self.cache.get_subfolders(symbol)

    def get_trained_dates(self, symbol, model):
        dates_string = self.cache.get_filenames(symbol, model)
        if dates_string:
            dates_string = dates_string[0] #in case of multiple files, we always use the first returned
            #filename format: "start_date__end_date.trm"
            start_date = dates_string.split('__')[0]
            end_date = dates_string.split('__')[1].split('.')[0]
            return (start_date, end_date)
        else:
            return (None, None)

    def run_forecast(self, symbol, model, date):
        """ Predict close prices for a stock symbol at a given date.
        
        Args:
            - symbol (str): stock symbol to build the model for.
            - model (PredictiveModel): predictive model to be used in testing.
            - date (Date): date at which prediction will take place.
        
        Returns:
            - return_plot (PlotInfo): plot data for last month data points + prediction
            - return_table(TableInfo): tabaulated results of prediction per forecast horizon
        """
        if symbol not in self.cache.get_folders():
            raise ValidationError('Symbol {} not found'.format(missing_symbols))
        dates = pd.date_range(end = date, periods = 365) #submit data of last year before prediction(252 samples)
        df = get_data_web([symbol], dates, False)
        training_summary = self.cache.load_single(symbol, model)
        if training_summary.empty: #file failed to load
            raise ValidationError('No training summary file found for the specified symbol and model.')
        num_models = 1

        return self.model.predict(df, training_summary, num_models)
