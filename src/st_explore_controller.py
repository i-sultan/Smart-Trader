"""Controller file for explore view and model
"""

#.-------------------.
#|      imports      |
#'-------------------'
from st_explore_model import *
from st_explore_view import *
from st_helpers import *

#.----------------.
#|    Classes     |
#'----------------'
class ExploreController(object):
    """ Controller class for explore model and view 

    The controller handles communications between model and view objects following
    the MVC design pattern.

    Args:
        - parent_view (tk widget): the view object which the explore view will
        be part of.

	"""
    def __init__(self, parent_view):
        self.model = ExploreModel()        
        self.view  = ExploreView(parent_view, self)

    def get_explorations(self):
        return [exploration.name for exploration in self.model.explorations_list]

    def run_exploration(self, id, symbols, start_date, end_date):
        """ Present an exploration for a stock symbol from start_date to end_date.
        
        Args:
            - id (int): ID of predictive model to be used in training.
            - symbol (str): stock symbol to build the model for.
            - start_date (Date): first date of training period.
            - end_date (Date): last date of training period.
        
        Returns:
            - exploration (PlotInfo): a plot of the requested exploration.
        """
        dates = pd.date_range(start_date, end_date)
        df = get_data_web(symbols, dates)
        return self.model.explorations_list[id].run(df, concat = True)