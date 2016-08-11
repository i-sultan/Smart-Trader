# Smart Trader System
ST system gives stock price prediction based on results from a training  phase, it searches through a large space of models and hyper-parameters, and present testing and validation results for any user selected symbol.

Input metrics (Indicators) and program details are explained in project documentation, the output is a prediction of adjusted close price.

## User Interface Description

Interface is split into:

* **Explore interface**:
   Presents visualizations of a given stock within a given period.

* **Training and Validation interface**:
   Produces validation results with different models and hyper-parametrs.

* **Test interface**:
   Uses the best model (in terms of MSE) found in training phase in order to predict and test close price within a given period

* **Forecast interface**:
   Gives prediction results at any single date without validation, this should be the only interface available to clients of the system.

## Project Structure

Project folders are structured as follows:

	cache/		- folder where all trained models, validation results, and testing results are saved.
	src/		- folder where all source code is saved
	README.txt	- this file
	Report.pdf	- PDF file of the final project report

Code of this project follows MVC (model, view, controller) design pattern for separation of responsibilities between GUI and core files.

The source folder is organized as follows:

	st_app.py					- Startup file, calls the main GUI view of the program.
	st_main_view.py				- Holds the initial GUI of the app and connects to different controllers and views in the project.

	st_cache_handler.py			- Class to facilitate interaction with the cache folder. Opens possibilities for other means of serving the trainsed models such as cloud, database, etc.
	st_helpers.py				- File for helper functions used by more than one class in ST system, like plotting functions, web retrieval, etc.

	st_backtest_controller.py	- Containts controller class for backtest model and view.
	st_explore_controller.py	- Containts controller class for explore model and view.
	st_forecast_controller.py	- Containts controller class for forecast model and view.
	st_train_controller.py		- Containts controller file for train view and train model.

	st_backtest_view.py			- Contains view class for backtesting.
	st_explore_view.py			- Contains view class for exploration.
	st_forecast_view.py			- Contains view class for forecasting.
	st_train_view.py			- Contains view class for training and validation.
	
	st_explore_model.py			- Contains model class for explore. Used to explore stocks.
	st_train_model.py			- Module for predictive models training, validation, and testing.

Code is developed with Python 2.7. Future versions (3.x) will be supported soon.


## External packages

The following external python packages need to be installed to run the code:
* matplotlib
* Tkinter, ttk
* pickle
* pandas
* numpy
* sklearn


## Running the code

To start the application, run st_app file from top application folder as follows:
python src/st_app.py

Notice: The application folder needs to be placed in unprotected location since new files will be written to the cache folder

st_app will display a user interface starting at the exploration tab, from there, below is a short sample tutorial of how to navigate through the system:

1. Enter a symbol for stock ("AAPL" for example)
2. Enter a starting date for exploration ("2016-01-01")
3. Enter an end date for exploration ("2016-07-01")
4. Select any exploration from the exploration types and hit the "Run Exploration" button.

![Explore GUI](/screenshots/explore.png)

In order to use testing or forecasting, a model needs to be trained first. To train a model, go to the "Train" tabe first, and follow the following example:

1. Enter a symbol for stock ("AAPL")
2. Enter a starting date for training ("2010-01-01")
3. Enter an end date for training ("2016-01-01")
4. Select any model from the predictive models list and hit the "Start Training" button.

Training progress will appear, and user can check the results of training and validation once training is done.

![Train GUI](/screenshots/train.png)

Next step is testing the model, testing ideally is done on a range that is outside testing range. 
The best model from all permutations in training step will be used in testing and forecasting, and hence the need for seperate periods. Otherwise, our testing results will not be indicative of the model generalization power.

To continue the above example, switch to "Backtest" tabe, and then:

1. select one of the pre-trained stocks ("AAPL")
2. Enter a starting date for testing ("2016-01-01")
3. Enter an end date for testing ("2016-07-01")
4. Select any model from the predictive models list and hit the "Run Backtesting" button.

Once testing is complete, a figure will show with results of different horizons. In addition, a csv file is saved in cache with the testing metrics results.

![Test GUI](/screenshots/test.png)

Forecasting is also done outside the training period. We expect the 'forecast' tab to be the only interface available to eventual customers, the following steps conclude our mini-tutorial:

1. select one of the pre-trained stocks ("AAPL")
 * Notice that training period is already filled based on the trained model.
2. Enter a prediction date ("2016-07-01") and choose whether to display a forecast plot with predictions added to last month results, or a forecast table containing prediction per forecast horizon.

![Forecast GUI](/screenshots/forecast.png)
