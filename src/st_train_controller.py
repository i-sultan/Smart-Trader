"""Controller file for train view and train model
"""

#.-------------------.
#|      imports      |
#'-------------------'
from st_train_model import *
from st_train_view import *
from st_helpers import *
from st_cache_handler import *

#.----------------.
#|    Classes     |
#'----------------'
class TrainingController(object):
    """ Controller class for training model and view. 
    
    The controller handles communications between model and view objects following
    the MVC design pattern.

    Args:
        - parent_view (tk widget): the view object which the backtest view will
        be part of.

	"""

    def __init__(self, parent_view):
        self.model = TrainingMaster()        
        self.view  = TrainingView(parent_view, self)
        self.cache = CacheHandler()

    def get_training_models(self):
        return [training_model.name for training_model in self.model.training_models]

    def run_training(self, id, symbol, start_date, end_date, update_progress):
        """ Train model on a stock symbol from start_date to end_date, and save the training summary.
        
        Args:
            - id (int): ID of predictive model to be used in training
            - symbol (str): stock symbol to build the model for
            - start_date (Date): first date of training period.
            - end_date (Date): last date of training period
            - progress (IntVar): percent progress of the training process
        
        Returns:
            - training_summary (TableInfo): containing model, validated params, and validation metrics
        """
        # add 1 year ahead of starting date for training
        dates = pd.date_range(start_date, end_date).union(pd.date_range(end = start_date, periods=365, freq='D'))
        df = get_data_web(symbol, dates, False)
        training_summary, training_presentation = self.model.train_model(df, self.model.training_models[id], update_progress)
        # save training summary at proper location
        folder_name = symbol[0]
        subfolder_name = self.model.training_models[id].name
        file_name = start_date + "__" + end_date + ".trm"
        self.cache.save_single(folder_name, subfolder_name, file_name, training_summary)

        file_name = start_date + "__" + end_date + ".csv"
        return self.cache.save_df(folder_name, subfolder_name, file_name, training_presentation)