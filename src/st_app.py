"""Main controller code for Smart Trader (ST) System implementation.

This is the startup file of the project, and includes the 
main controller which calls the main GUI view of the program.

ST system gives stock price prediction based on results from a training 
phase, it searches through a large space of models and hyper-parameters,
and present testing and validation results for any user selected symbol.

Input metrics (Indicators) and program details are explained in project
documentation, the output is a prediction of adjusted close price.

Interface is split into:

* Explore interface:
   Presents visualizations of a given stock within a given period.

* Training and Validation interface:
   Produces validation results with different models and hyper-parametrs.

* Test interface:
   Uses the best model (in terms of MSE) found in training phase in order
   to predict and test close price within a given period

* Forecast interface:
   Gives prediction results at any single date without validation, this
   should be the only interface available to clients of the system.
"""

#.-------------------.
#|      imports      |
#'-------------------'
from st_main_view import *

#.-------------------.
#|    main entry     |
#'-------------------'
def main():    
    app = MainView()
    app.mainloop()

if __name__ == '__main__':
    main()