"""Module for predictive models training, validation, and testing.
"""

#.-------------------.
#|      imports      |
#'-------------------'
import time
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import itertools
from st_helpers import *
from copy import deepcopy
#k-NN
from sklearn.neighbors import KNeighborsRegressor
#Least Squares
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
#SVM
from sklearn import svm
from sklearn import preprocessing
#Random Forest
from sklearn.ensemble import RandomForestRegressor
#ADA Boost
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
#Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

#.-----------------.
#|    constants    |
#'-----------------'
h=[1,5,10,21] #forecast horizon values

#.-------------------------.
#|    helper functions     |
#'-------------------------'
def prepare_samples(df, model_indicators):
    """ Prepare a numpy array of samples based on a samples dataframe and dicitonary of indicators.
    
    Args:
        - df (DataFrame): dataframe of samples with indicators as columns .
        - model_indicators (dictionary): dictionary of indicators and number of historical samples 
            per indicator used in a predictive model.
        
    Returns:
        - samples (2D numpy array): where each row is a single sample with indicators as columns.
        - labels (list of 1D numpy arrays): containing the proper result per sample for each forecast horizon
            where length of list is the number of forecast horizons.
    """
    df=df.rename(columns = {df.columns[0]:'Daily'}) # first column in df must be daily returns observations, rest are additional indicators. 

    first_sample = np.max(model_indicators.values())
    indicators_per_sample = sum(model_indicators.values())
    total_samples = df.shape[0]
     
    samples = np.array([])
    labels = [np.array([]) for _ in h]
    daily_index = 0
    
    for i in range(total_samples - first_sample):
        for key, value in model_indicators.iteritems():
            if key == 'Daily': # keep daily values at the beginning
                samples = np.insert(samples,daily_index,df.ix[first_sample-value+i:first_sample+i, key])
            else:
                samples = np.append(samples,df.ix[first_sample-value+i:first_sample+i, key])
        daily_index = len(samples)

        for h_index in range(len(h)):
            if i + h[h_index] <= total_samples - first_sample:
                labels[h_index] = np.append(labels[h_index], df.ix[h[h_index]+first_sample+i-1, 0])

    return (first_sample, samples.reshape(-1, indicators_per_sample), labels)

def prepare_last_sample(df, model_indicators):
    """ Prepare a numpy array of last sample based on a samples dataframe and dicitonary of indicators.
    
    Args:
        - df (DataFrame): dataframe of samples with indicators as columns 
        - model_indicators (dictionary): dictionary of indicators and number of previous samples 
            per indicator used in a predictive model
        
    Returns:
        - sample (1D numpy array): a single sample with indicators as columns 
    """
    df=df.rename(columns = {df.columns[0]:'Daily'}) # first column in df must be daily returns observations, rest are additional indicators. 

    indicators_per_sample = sum(model_indicators.values())
    total_samples = df.shape[0]
     
    sample = np.array([])
    for key, value in model_indicators.iteritems():
        if key == 'Daily': # keep daily values at the beginning
            sample = np.insert(sample,0,df.ix[total_samples-value:total_samples, key])
        else:
            sample = np.append(sample,df.ix[total_samples-value:total_samples, key])

    return sample.reshape(1, indicators_per_sample)

def apply_exp_weights(samples, exp_weight):
    """ Apply exponential weights to input samples. """
    weights = [exp_weight**(index-1) for index in range(samples.shape[1],0,-1)]
    return samples * weights

def set_presentation(samples, labels, sample_presentation, num_samples):
    """ Set samples presentation according to selected format. """
    if(sample_presentation == SamplePresentation.absolute):
        return (samples, [np.reshape(labels[h_index],(-1,1)) for h_index in range(len(h))], [1]*samples.shape[0])
    elif (sample_presentation == SamplePresentation.cumulative):
        normalizer = np.reshape(samples[:,num_samples-1].copy(),(-1,1))
        presented_samples = samples.copy()
        presented_samples[:,:num_samples] = samples[:,:num_samples]/normalizer-np.ones((samples.shape[0],1))
        presented_labels = [0 for i in range (len(h))]
        for h_index in range(len(h)):
            len_labels = len(labels[h_index])
            presented_labels[h_index] = np.reshape(labels[h_index],(-1,1))/normalizer[:len_labels]-np.ones((len_labels,1))
        return presented_samples, presented_labels, normalizer
    elif (sample_presentation == SamplePresentation.daily):
        normalizer = np.reshape(samples[:,num_samples-1].copy(),(-1,1))
        shifted_samples =np.append(samples[:,:num_samples], normalizer, axis = 1)[:,1:]
        presented_samples = samples.copy()
        presented_samples[:,:num_samples] = samples[:,:num_samples]/shifted_samples - np.ones((len(normalizer),1))
        presented_labels = [0 for i in range (len(h))]
        for h_index in range(len(h)):
            len_labels = len(labels[h_index])
            presented_labels[h_index] = np.reshape(labels[h_index],(-1,1))/normalizer[:len_labels]-np.ones((len_labels,1))
        return presented_samples, presented_labels, normalizer

def remove_presentation(value, normalizer, sample_presentation):
    """ Remove sample presentation according to selected format in order to get absolute value. """
    if(sample_presentation == SamplePresentation.absolute):
        return value
    else:
        return (value+1)*normalizer

def summarize(predictive_model, trained_model, error_list, relative_error_list, hit_list, params, total_train_time, total_test_time):
    """ Generate a summary DataFrame containing a trained predictive model and its testing metrics. """
    return_df = pd.DataFrame()
    predictive_model.pretrained_model = trained_model
    for h_index in range(len(h)):
        MSE, RMSE, MAE, MAPE, hit_rate = calc_errors(error_list[h_index], 
                                                     relative_error_list[h_index], hit_list[h_index])
        return_df = return_df.append({'trained_model':deepcopy(predictive_model), 'horizon': h[h_index], 'params' : params, 
                'MSE' : MSE, 'RMSE' : RMSE, 'MAE' : MAE, 'MAPE' : MAPE, 'hit_rate' : hit_rate,
                'training_time' : total_train_time[h_index], 'testing_time' : total_test_time[h_index]}, ignore_index = True)
    return return_df

def create_metrics_df(model_name, error_list, relative_error_list, hit_list):
    """ Create a metrics DataFrame from erros lists and hit list. """
    MSE, RMSE, MAE, MAPE, hit_rate = calc_errors(error_list, relative_error_list, hit_list)
    return_df = pd.DataFrame({'MSE' : MSE, 'RMSE' : RMSE, 'MAE' : MAE, 'MAPE' : MAPE, 'hit_rate' : hit_rate}, index = [model_name])
    return return_df

def make_presentable(training_summary, model_name):
    """ Cleanup training summary DataFrame in order to present it to user. """
    training_presentation = training_summary.copy()
    training_presentation['trained_model'] = model_name
    training_presentation.set_index('trained_model', inplace = True)
    return training_presentation

def pick_best(training_summary, horizon, num_models, unique = False):
    """ Select best num_models based on MSE from training summary."""
    training_summary.sort_values(by = 'MSE', ascending = True, inplace = True)
    if unique:
        training_summary = training_summary.drop_duplicates(subset = ['MSE'])
    return training_summary[training_summary['horizon']==horizon].reset_index().ix[0:num_models-1]

#.----------------.
#|    Classes     |
#'----------------'
class TrainingMaster(object):
    """ Model class for training. Trains different predictive models on stocks data.
    
    Model class executes requests received from the controller following
    the MVC design pattern.

    TrainingMaster presents all predictive models to training controller, and handles
    user requests after that.

    """

    def __init__(self):
        """ Initialize internal and external parameters."""    
        #instantiate and list training models
        knn = Knn()
        ls  = LeastSquares()
        svr = SVReg()
        rf  = RandomForest()
        ab  = AdaBoost()
        gb  = GradientBoost()
        sg  = StackedGeneralization()
        self.training_models = [knn, ls, svr, rf, ab, gb, sg]

    def train_model(self, df, model, update_progress):
        """ Select validation samples from input DataFram, then run selected predictive model.
        
        Args:
            - df (DataFrame): dataframe of samples with indicators as columns.
            - model (PredictiveModel): predictive model to train and validate.
            - progress (IntVar): holds percent progress of the training process.
        
        Returns:
            - training_summary (DataFrame): contains model, validated hyper-params, and validation metrics
        """
        validation_start = 252 #index of the first validation point
        validation_iterations = 20
        test_size = 100 #number of samples left out for testing

        validation_end = df.shape[0]-test_size
        validation_step = (validation_end - validation_start)/validation_iterations
        
        if validation_end > validation_start:
            validation_range = range(validation_start, validation_end, validation_step)
        else:
            validation_range = []
        testing_range = range(validation_end,df.shape[0])
        
        training_summary = model.train_validate(df, validation_range + testing_range, update_progress)
        return training_summary

class ModelsCollection(object):
    """ Class that picks best models for training and prediction based on a training summary."""

    def __init__(self):
        pass

    def backtest(self, df, training_summary, num_models):
        """ Test all points in df using best models and return a dataframe with predicted and actual values.
        
        Args:
            - df (DataFrame): dataframe of samples with indicators as columns, notice: first prediction 
                point follows the last sample in df.
            - training_summary (DataFrame): containing model, validated params, and validation metrics.
            - num_models (int): number of best models to select to optionaly form an averaging ensemble.
        
        Returns:
            - return_plot (PlotInfo): plot containing actual data against forecast at all horizons.
            - metrics_df (DataFrame): dataframe contiaining testing metrics.
        """
        metrics_df = pd.DataFrame()
        return_df  = pd.DataFrame()
        for h_index in range(len(h)):
            training_models = pick_best(training_summary, h[h_index], num_models)
            for _, training_model in training_models.iterrows():
                backtest_df = training_model['trained_model'].backtest(training_model['params'], h_index, df) 
                # merge only new columns from backtest dataframe
                new_columns = [colname for colname in backtest_df.columns.values if colname not in return_df.columns.values]
                return_df = pd.concat([return_df, backtest_df.ix[:,new_columns]], axis = 1)
        
            #calculate metrics
            skipped_samples = 42 #drop first two months from metrics calculation, not enough training
            reference = return_df.ix[skipped_samples:,0] 
            error_list = (return_df.ix[skipped_samples:,-1] - reference).tolist() 
            relative_error_list = ((return_df.ix[skipped_samples:,-1] - reference)/reference).tolist()
            original_data = return_df.ix[skipped_samples-h[h_index]:,0].values[:-h[h_index]]
            hit_list   = ((return_df.ix[skipped_samples:,-1].values - original_data)*(reference.values - original_data)>0)
            metrics_df = metrics_df.append(create_metrics_df(return_df.columns[-1], error_list, relative_error_list, hit_list))

        return_df['new_index'] = df.index[-return_df.shape[0]:] #retain the date index from original df
        return_df = return_df.set_index('new_index')
        return_plot = PlotInfo(return_df, 'Backtest result vs. date','Date',
                               'Price',type = PlotTypes.Plot, line_styles = ['-'] + ['--']*(len(h)*num_models))
        return return_plot, metrics_df

    def predict(self, df, training_summary, num_models):
        """ Predict point following the last sample in df.
        
        Args:
            - df (DataFrame): dataframe of samples with indicators as columns, notice: first prediction 
                point follows the last sample in df 
            - training_summary (DataFrame): containing model, validated params, and validation metrics
            - num_models (int): number of models to select forming an ensemble
        
        Returns:
            - return_plot (PlotInfo): plot data for last month data points + prediction
            - return_table(TableInfo): tabaulated results of prediction per forecast horizon
        """
        predictions = []
        for h_index in range(len(h)):
            result = []
            training_models = pick_best(training_summary, h[h_index], num_models)
            for _, training_model in training_models.iterrows():
                result.append(training_model['trained_model'].predict(training_model['params'], h_index, df))
            predictions.append(np.mean(result))
        
        #fix the index of prediction dataframe before appending
        predictions_timeline_df = pd.DataFrame(np.reshape(predictions,(-1,1)), columns = ['Prediction'])
        us_buisness_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        prediction_dates = [0]*len(h)
        for h_index in range(len(h)):
            prediction_dates[h_index] = df.index[-1]+ h[h_index]*us_buisness_day
        predictions_timeline_df['new_index'] = prediction_dates
        predictions_timeline_df = predictions_timeline_df.set_index('new_index')

        return_plot = PlotInfo(df.iloc[-21:,0].append(predictions_timeline_df), #append predictions to last month data
                               'Last month data + prediction vs. date','Date',
                               'Price',type = PlotTypes.Plot, line_styles = ['-','o'])

        #predictions_df = pd.DataFrame(np.reshape(h+predictions,(2,-1)).T, columns = ['Forecast Horizon','Prediction'])
        #predictions_df.set_index('Forecast Horizon')
        predictions_df = pd.DataFrame(np.reshape(predictions,(-1,1)), columns = ['Prediction'])
        predictions_df.index = ['h = '+str(h[h_index]) for h_index in range(len(h))]
        return_table= TableInfo(predictions_df, "Prediction Results")
        return return_plot, return_table

class SamplePresentation(object):
    absolute, daily, cumulative = "A", "D", "C"

class PredictiveModel(object):
    """ Training model class wraps methods relevant to predictive models."""
    def __init__(self):
        """ Initialize predictive model with model, model indicators, and params. """
        pass
    def train_validate(self, df, validation_range, update_progress):
        """ Train and validate regressor on df samples with indices listed in validation_range. """
        pass
    def backtest(self, params, h_index, df, initialization_samples):
        """ Train and validate regressor on all samples in dataframe. """
        pass
    def predict(self, params, h_index, df):
        """ Forecast beyond last sample in the dataframe using pretrained model. """
        pass


class Knn(PredictiveModel):
    """ k-NN regressive predictive model."""

    def __init__(self):
        """ Initialize predictive model with model, model indicators, and params. """
        self.name = "k-Nearest Neighbors"
        self.summary_name = "k-NN"
        self.indicators_samples = {'Daily':42}
        self.model_params = dict( k          = [1, 2, 4, 8, 16],
                                  exp_weight = [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                                  sample_presentation = [SamplePresentation.daily, 
                                                         SamplePresentation.absolute, SamplePresentation.cumulative])
        self.pretrained_model = None #save the pretrained model for future use

    def train_validate(self, df, validation_range, update_progress):
        """ Train and validate regressor on df samples with indices listed in validation_range. """
        training_summary = pd.DataFrame()
        first_sample, samples, labels = prepare_samples(df, self.indicators_samples)

        # progress bar parameters
        total_steps = len(self.model_params['sample_presentation']) * \
                      len(self.model_params['exp_weight']) * len(self.model_params['k'])
        completed_steps = 0

        # loop over model parameters
        for sample_presentation in self.model_params['sample_presentation']:
            presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, self.indicators_samples['Daily'])

            for exp_weight in self.model_params['exp_weight']:
                weighted_samples = apply_exp_weights(presented_samples, exp_weight)

                for k in self.model_params['k']:
                    model, total_train_time, total_test_time = [[0 for i in range (len(h))] for j in range(3)]
                    error_list, relative_error_list, hit_list = [[[] for i in range (len(h))] for j in range(3)]
                    params = (sample_presentation, exp_weight, k)

                    # model training and validation core
                    for h_index in range(len(h)):
                        for index in validation_range:
                            i = index-first_sample                        
                            x_train, x_validate = weighted_samples[:i-h[h_index]+1,:], weighted_samples[i,:] #need to stop training h steps before test
                            y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                            #train
                            t1 = time.time()
                            model[h_index] = KNeighborsRegressor(n_neighbors=k) # train a separate model for each horizon
                            model[h_index].fit(x_train, y_train)
                            t2 = time.time()
                            train_time = (t2-t1)
                            #test
                            y_predict = model[h_index].predict(x_validate.reshape(1,-1))
                            test_time = (time.time()-t2)
                            #apend new results
                            y_validate_absolute = remove_presentation(y_validate,normalizer[i], sample_presentation)
                            y_predict_absolute  = remove_presentation(y_predict ,normalizer[i], sample_presentation)
                            error_list[h_index] += [y_validate_absolute - y_predict_absolute]
                            relative_error_list[h_index] += [(y_validate_absolute - y_predict_absolute)/y_validate_absolute]
                            hit_list[h_index] += [(y_validate-x_validate[-1])*(y_predict-x_validate[-1]) > 0]
            
                            total_train_time[h_index] += train_time
                            total_test_time[h_index] += test_time
                            if i == len(presented_labels[h_index])-1:
                                #very last training point, include last training oppurtunity
                                x_train = weighted_samples[:i+1,:]
                                y_train = presented_labels[h_index][:i+1]
                                model[h_index].fit(x_train, y_train)
                                break
                    
                    completed_steps += 1
                    update_progress(100.0 * completed_steps/total_steps)

                    #save last trained model, and add to training summary
                    training_summary = training_summary.append(summarize(self, model, error_list, relative_error_list, hit_list, 
                                                                        params, total_train_time, total_test_time))
        return training_summary, make_presentable(training_summary, self.summary_name)

    def backtest(self, (sample_presentation, exp_weight, k), h_index, df, initialization_samples = 1250):
        """ Train and validate regressor on all samples in dataframe. """
        first_sample, samples, labels = prepare_samples(df, self.indicators_samples)

        presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, self.indicators_samples['Daily'])
        weighted_samples = apply_exp_weights(presented_samples, exp_weight)
        model, y_predict = [[0 for i in range (len(h))] for j in range(2)]
        return_y = []

        for i in range(initialization_samples - max(h) + 1, df.shape[0]): #first 5 years is bootstrapping samples only
            i -= first_sample
            if i>=len(presented_labels[h_index]):
                y_predict = None
            else:
                x_train, x_validate = weighted_samples[:i-h[h_index]+1,:], weighted_samples[i,:] #need to stop training h steps before test
                y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                #train
                model = KNeighborsRegressor(n_neighbors=k) # train a separate model for each horizon
                model.fit(x_train, y_train)
                #predict
                y_predict = model.predict(x_validate.reshape(1,-1))[0]
                y_predict = remove_presentation(y_predict, normalizer[i], sample_presentation)[0]

            return_y.append([labels[0][i], y_predict])
        colnames = ['actual'] + \
                   [self.summary_name+str((sample_presentation, exp_weight, k))+', h='+ str(h[h_index])]
        return_df= pd.DataFrame(return_y, columns = colnames)
        #shift each column forward by h to align timelines
        return_df.ix[:,1] = return_df.ix[:,1].shift(h[h_index]-1) #first column is 'actual' and need not be shifted
        return_df = return_df._ix[max(h)-1:,:]
        return return_df

    def predict(self, (sample_presentation, exp_weight, k), h_index, df):
        """ Forecast beyond last sample in the dataframe using pretrained model. """
        sample = prepare_last_sample(df, self.indicators_samples)
        dummy_labels = [[0]]*len(h)
        presented_sample, _, normalizer = set_presentation(sample, dummy_labels, sample_presentation, self.indicators_samples['Daily'])
        weighted_sample = apply_exp_weights(presented_sample, exp_weight)

        prediction = self.pretrained_model[h_index].predict(weighted_sample.reshape(1,-1))[0]
        prediction = remove_presentation(prediction, normalizer[-1], sample_presentation)
        return prediction


class LeastSquares(PredictiveModel):
    """ Least squares based regressive predictive model with elastic nets."""

    def __init__(self):
        """ Initialize predictive model with model, model indicators, and params. """
        self.name = "Least squares based"
        self.summary_name = "LS"
        self.indicators_samples      = {'Daily':42}
        self.full_indicators_samples = {'Daily':42, 'Volume':10, 'Open':10, 'High':10, 'Low':10, 'SMA':5, 'EWMA':5, 'MOM':5, 'STD':5}
        self.model_params = dict(degree = [1],
                                 alpha  = [1, 0.5, 0],
                                 rho    = [1, 0.5, 0],
                                 full_indicators     = [True, False],
                                 sample_presentation = [SamplePresentation.cumulative])
        self.pretrained_model = None #save the pretrained model for future use

    def train_validate(self, df, validation_range, update_progress):
        """ Train and validate regressor on df samples with indices listed in validation_range. """
        training_summary = pd.DataFrame()

        # progress bar parameters
        total_steps = len(self.model_params['full_indicators']) * \
                      len(self.model_params['sample_presentation']) * len(self.model_params['degree']) * \
                      len(self.model_params['alpha'])
        completed_steps = 0

        # loop over model parameters
        for use_full_indicators in self.model_params['full_indicators']:
            if use_full_indicators:
                first_sample, samples, labels = prepare_samples(df, self.full_indicators_samples)
                daily_samples = self.full_indicators_samples['Daily']
            else:
                first_sample, samples, labels = prepare_samples(df, self.indicators_samples)
                daily_samples = self.indicators_samples['Daily']

            for sample_presentation in self.model_params['sample_presentation']:
                presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, daily_samples)
                for degree in self.model_params['degree']:
                    for alpha in self.model_params['alpha']:
                        for rho in self.model_params['rho']:
                            model, total_train_time, total_test_time = [[0 for i in range (len(h))] for j in range(3)]
                            error_list, relative_error_list, hit_list = [[[] for i in range (len(h))] for j in range(3)]
                            params = (sample_presentation, use_full_indicators, degree, alpha, rho)

                            # model training and validation core
                            for h_index in range(len(h)):
                                for index in validation_range:
                                    i = index-first_sample                        
                                    x_train, x_validate = presented_samples[:i-h[h_index]+1,:], presented_samples[i,:] #need to stop training h steps before test
                                    y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                                    #train
                                    t1 = time.time()
                                    if alpha == 0:
                                        model[h_index] = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                                    elif rho == 0:
                                        model[h_index] = make_pipeline(PolynomialFeatures(degree), Ridge(alpha = alpha))
                                    elif rho == 1:
                                        model[h_index] = make_pipeline(PolynomialFeatures(degree), Lasso(alpha = alpha))
                                    else:
                                        model[h_index] = make_pipeline(PolynomialFeatures(degree), ElasticNet(alpha = alpha, l1_ratio = rho))
                                    model[h_index].fit(x_train, y_train)
                                    t2 = time.time()
                                    train_time = (t2-t1)
                                    #test
                                    y_predict = model[h_index].predict(x_validate.reshape(1,-1))
                                    test_time = (time.time()-t2)
                                    #apend new results
                                    y_validate_absolute = remove_presentation(y_validate,normalizer[i], sample_presentation)
                                    y_predict_absolute  = remove_presentation(y_predict ,normalizer[i], sample_presentation)
                                    error_list[h_index] += [y_validate_absolute - y_predict_absolute]
                                    relative_error_list[h_index] += [(y_validate_absolute - y_predict_absolute)/y_validate_absolute]
                                    hit_list[h_index] += [(y_validate-x_validate[-1])*(y_predict-x_validate[-1]) > 0]
            
                                    total_train_time[h_index] += train_time
                                    total_test_time[h_index] += test_time
                                    if i == len(presented_labels[h_index])-1:
                                        #very last training point, include last training oppurtunity
                                        x_train = presented_samples[:i+1,:]
                                        y_train = presented_labels[h_index][:i+1]
                                        model[h_index].fit(x_train, y_train)
                                        break
                            #save last trained model, and add to training summary
                            training_summary = training_summary.append(summarize(self, model, error_list, relative_error_list, hit_list, 
                                                                                params, total_train_time, total_test_time))

                        completed_steps += 1
                        update_progress(100.0 * completed_steps/total_steps)
        return training_summary, make_presentable(training_summary, self.summary_name)

    def backtest(self, (sample_presentation, use_full_indicators, degree, alpha, rho), h_index, df, initialization_samples = 1250):
        """ Train and validate regressor on all samples in dataframe. """
        if use_full_indicators:
            first_sample, samples, labels = prepare_samples(df, self.full_indicators_samples)
            daily_samples = self.full_indicators_samples['Daily']
        else:
            first_sample, samples, labels = prepare_samples(df, self.indicators_samples)
            daily_samples = self.indicators_samples['Daily']

        presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, daily_samples)
        model, y_predict = [[0 for i in range (len(h))] for j in range(2)]
        return_y = []

        for i in range(initialization_samples - max(h) + 1, df.shape[0]): #first 5 years is bootstrapping samples only
            i -= first_sample
            if i>=len(presented_labels[h_index]):
                y_predict = None
            else:
                x_train, x_validate = presented_samples[:i-h[h_index]+1,:], presented_samples[i,:] #need to stop training h steps before test
                y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                #train
                if alpha == 0:
                    model = make_pipeline(PolynomialFeatures(degree), LinearRegression(normalize = True))
                elif rho == 0:
                    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha = alpha, normalize = True, max_iter = 10000))
                elif rho == 1:
                    model = make_pipeline(PolynomialFeatures(degree), Lasso(alpha = alpha, normalize = True, max_iter = 10000))
                else:
                    model = make_pipeline(PolynomialFeatures(degree), ElasticNet(alpha = alpha, l1_ratio = rho, normalize = True, max_iter = 10000))
                model.fit(x_train, y_train)
                #predict
                y_predict = model.predict(x_validate.reshape(1,-1))[0]
                y_predict = remove_presentation(y_predict, normalizer[i], sample_presentation)[0]

            return_y.append([labels[0][i], y_predict])
        colnames = ['actual'] + \
                   [self.summary_name+str((sample_presentation, use_full_indicators, degree, alpha, rho))+', h='+ str(h[h_index])]
        return_df= pd.DataFrame(return_y, columns = colnames)
        #shift each column forward by h to align timelines
        return_df.ix[:,1] = return_df.ix[:,1].shift(h[h_index]-1) #first column is 'actual' and need not be shifted
        return_df = return_df._ix[max(h)-1:,:]
        return return_df

    def predict(self, (sample_presentation, use_full_indicators, degree, alpha, rho), h_index, df):
        """ Forecast beyond last sample in the dataframe using pretrained model. """
        if use_full_indicators:
            sample = prepare_last_sample(df, self.full_indicators_samples)
            daily_samples = self.full_indicators_samples['Daily']
        else:
            sample = prepare_last_sample(df, self.indicators_samples)
            daily_samples = self.indicators_samples['Daily']
        
        dummy_labels = [[0]]*len(h)
        presented_sample, _, normalizer = set_presentation(sample, dummy_labels, sample_presentation, daily_samples)

        prediction = self.pretrained_model[h_index].predict(presented_sample.reshape(1,-1))[0]
        prediction = remove_presentation(prediction, normalizer[-1], sample_presentation)
        return prediction


class SVReg(PredictiveModel):
    """ Support vector regression predictive model."""

    def __init__(self):
        """ Initialize predictive model with model, model indicators, and params. """
        self.name = "Support Vector"
        self.summary_name = "SVR"
        self.indicators_samples      = {'Daily':42}
        self.full_indicators_samples = {'Daily':42, 'Volume':10, 'Open':10, 'High':10, 'Low':10, 'SMA':5, 'EWMA':5, 'MOM':5, 'STD':5}
        self.model_params = dict(kernel    = ['poly', 'rbf'],
                                 C         = [1e-2, 0.1, 1, 10],
                                 tolerance = [.001, 0.1],
                                 full_indicators     = [True, False],
                                 sample_presentation = [SamplePresentation.cumulative])
        self.pretrained_model = None #save the pretrained model for future use

    def train_validate(self, df, validation_range, update_progress):
        """ Train and validate regressor on df samples with indices listed in validation_range. """
        training_summary = pd.DataFrame()

        # progress bar parameters
        total_steps = len(self.model_params['full_indicators']) * \
                      len(self.model_params['sample_presentation']) * len(self.model_params['kernel']) * \
                      len(self.model_params['C'])
        completed_steps = 0

        # loop over model parameters
        for use_full_indicators in self.model_params['full_indicators']:
            if use_full_indicators:
                first_sample, samples, labels = prepare_samples(df, self.full_indicators_samples)
                daily_samples = self.full_indicators_samples['Daily']
            else:
                first_sample, samples, labels = prepare_samples(df, self.indicators_samples)
                daily_samples = self.indicators_samples['Daily']

            for sample_presentation in self.model_params['sample_presentation']:
                presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, daily_samples)
                for kernel in self.model_params['kernel']:
                    for C in self.model_params['C']:
                        for tolerance in self.model_params['tolerance']:
                            model, total_train_time, total_test_time = [[0 for i in range (len(h))] for j in range(3)]
                            error_list, relative_error_list, hit_list = [[[] for i in range (len(h))] for j in range(3)]
                            params = (sample_presentation, use_full_indicators, kernel, C, tolerance)

                            # model training and validation core
                            for h_index in range(len(h)):
                                for index in validation_range:
                                    i = index-first_sample                        
                                    x_train, x_validate = presented_samples[:i-h[h_index]+1,:], presented_samples[i,:] #need to stop training h steps before test
                                    y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                                    #train
                                    t1 = time.time()
                                    model[h_index] = make_pipeline(preprocessing.StandardScaler(), svm.SVR(kernel = kernel, C = C, tol = tolerance, gamma = 'auto'))
                                    model[h_index].fit(x_train, np.ravel(y_train))
                                    t2 = time.time()
                                    train_time = (t2-t1)
                                    #test
                                    y_predict = model[h_index].predict(x_validate.reshape(1,-1))
                                    test_time = (time.time()-t2)
                                    #apend new results
                                    y_validate_absolute = remove_presentation(y_validate,normalizer[i], sample_presentation)
                                    y_predict_absolute  = remove_presentation(y_predict ,normalizer[i], sample_presentation)
                                    error_list[h_index] += [y_validate_absolute - y_predict_absolute]
                                    relative_error_list[h_index] += [(y_validate_absolute - y_predict_absolute)/y_validate_absolute]
                                    hit_list[h_index] += [(y_validate-x_validate[-1])*(y_predict-x_validate[-1]) > 0]
            
                                    total_train_time[h_index] += train_time
                                    total_test_time[h_index] += test_time
                                    if i == len(presented_labels[h_index])-1:
                                        #very last training point, include last training oppurtunity
                                        x_train = presented_samples[:i+1,:]
                                        y_train = presented_labels[h_index][:i+1]
                                        model[h_index].fit(x_train, np.ravel(y_train))
                                        break

                            #save last trained model, and add to training summary
                            training_summary = training_summary.append(summarize(self, model, error_list, relative_error_list, hit_list, 
                                                                                params, total_train_time, total_test_time))

                        completed_steps += 1
                        update_progress(100.0 * completed_steps/total_steps)
        return training_summary, make_presentable(training_summary, self.summary_name)

    def backtest(self, (sample_presentation, use_full_indicators, kernel, C, tolerance), h_index, df, initialization_samples = 1250):
        """ Train and validate regressor on all samples in dataframe. """
        if use_full_indicators:
            first_sample, samples, labels = prepare_samples(df, self.full_indicators_samples)
            daily_samples = self.full_indicators_samples['Daily']
        else:
            first_sample, samples, labels = prepare_samples(df, self.indicators_samples)
            daily_samples = self.indicators_samples['Daily']

        presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, daily_samples)
        model, y_predict = [[0 for i in range (len(h))] for j in range(2)]
        return_y = []

        for i in range(initialization_samples - max(h) + 1, df.shape[0]): #first 5 years is bootstrapping samples only
            i -= first_sample
            if i>=len(presented_labels[h_index]):
                y_predict = None
            else:
                x_train, x_validate = presented_samples[:i-h[h_index]+1,:], presented_samples[i,:] #need to stop training h steps before test
                y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                #train
                model = make_pipeline(preprocessing.StandardScaler(), svm.SVR(kernel = kernel, C = C, tol = tolerance, gamma = 'auto'))
                model.fit(x_train, np.ravel(y_train))
                #predict
                y_predict = model.predict(x_validate.reshape(1,-1))[0]
                y_predict = remove_presentation(y_predict, normalizer[i], sample_presentation)[0]

            return_y.append([labels[0][i], y_predict])
        colnames = ['actual'] + \
                   [self.summary_name+str((sample_presentation, use_full_indicators, kernel, C, tolerance))+', h='+ str(h[h_index])]
        return_df= pd.DataFrame(return_y, columns = colnames)
        #shift each column forward by h to align timelines
        return_df.ix[:,1] = return_df.ix[:,1].shift(h[h_index]-1) #first column is 'actual' and need not be shifted
        return_df = return_df._ix[max(h)-1:,:]
        return return_df

    def predict(self, (sample_presentation, use_full_indicators, degree, alpha, rho), h_index, df):
        """ Forecast beyond last sample in the dataframe using pretrained model. """
        if use_full_indicators:
            sample = prepare_last_sample(df, self.full_indicators_samples)
            daily_samples = self.full_indicators_samples['Daily']
        else:
            sample = prepare_last_sample(df, self.indicators_samples)
            daily_samples = self.indicators_samples['Daily']
        
        dummy_labels = [[0]]*len(h)
        presented_sample, _, normalizer = set_presentation(sample, dummy_labels, sample_presentation, daily_samples)

        prediction = self.pretrained_model[h_index].predict(presented_sample.reshape(1,-1))[0]
        prediction = remove_presentation(prediction, normalizer[-1], sample_presentation)
        return prediction


class RandomForest(PredictiveModel):
    """ Support vector regression predictive model."""

    def __init__(self):
        """ Initialize predictive model with model, model indicators, and params. """
        self.name = "Random Forest"
        self.summary_name = "RF"
        self.indicators_samples      = {'Daily':42}
        self.full_indicators_samples = {'Daily':42, 'Volume':10, 'Open':10, 'High':10, 'Low':10, 'SMA':5, 'EWMA':5, 'MOM':5, 'STD':5}
        self.model_params = dict(n_estimators = [5,10,20],
                                 max_features = ['sqrt', 'log2'],
                                 min_samples_split   = [2,4,8],
                                 full_indicators     = [True, False],
                                 sample_presentation = [SamplePresentation.cumulative])
        self.pretrained_model = None #save the pretrained model for future use

    def train_validate(self, df, validation_range, update_progress):
        """ Train and validate regressor on df samples with indices listed in validation_range. """
        training_summary = pd.DataFrame()

        # progress bar parameters
        total_steps = len(self.model_params['full_indicators']) * \
                      len(self.model_params['sample_presentation']) * len(self.model_params['n_estimators']) * \
                      len(self.model_params['max_features'])
        completed_steps = 0

        # loop over model parameters
        for use_full_indicators in self.model_params['full_indicators']:
            if use_full_indicators:
                first_sample, samples, labels = prepare_samples(df, self.full_indicators_samples)
                daily_samples = self.full_indicators_samples['Daily']
            else:
                first_sample, samples, labels = prepare_samples(df, self.indicators_samples)
                daily_samples = self.indicators_samples['Daily']

            for sample_presentation in self.model_params['sample_presentation']:
                presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, daily_samples)
                for n_estimators in self.model_params['n_estimators']:
                    for max_features in self.model_params['max_features']:
                        for min_samples_split in self.model_params['min_samples_split']:
                            model, total_train_time, total_test_time = [[0 for i in range (len(h))] for j in range(3)]
                            error_list, relative_error_list, hit_list = [[[] for i in range (len(h))] for j in range(3)]
                            params = (sample_presentation, use_full_indicators, n_estimators, max_features, min_samples_split)

                            # model training and validation core
                            for h_index in range(len(h)):
                                for index in validation_range:
                                    i = index-first_sample                        
                                    x_train, x_validate = presented_samples[:i-h[h_index]+1,:], presented_samples[i,:] #need to stop training h steps before test
                                    y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                                    #train
                                    t1 = time.time()
                                    model[h_index] = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators = n_estimators, 
                                                                                                                         max_features = max_features, min_samples_split = min_samples_split))
                                    model[h_index].fit(x_train, np.ravel(y_train))
                                    t2 = time.time()
                                    train_time = (t2-t1)
                                    #test
                                    y_predict = model[h_index].predict(x_validate.reshape(1,-1))
                                    test_time = (time.time()-t2)
                                    #apend new results
                                    y_validate_absolute = remove_presentation(y_validate,normalizer[i], sample_presentation)
                                    y_predict_absolute  = remove_presentation(y_predict ,normalizer[i], sample_presentation)
                                    error_list[h_index] += [y_validate_absolute - y_predict_absolute]
                                    relative_error_list[h_index] += [(y_validate_absolute - y_predict_absolute)/y_validate_absolute]
                                    hit_list[h_index] += [(y_validate-x_validate[-1])*(y_predict-x_validate[-1]) > 0]
            
                                    total_train_time[h_index] += train_time
                                    total_test_time[h_index] += test_time
                                    if i == len(presented_labels[h_index])-1:
                                        #very last training point, include last training oppurtunity
                                        x_train = presented_samples[:i+1,:]
                                        y_train = presented_labels[h_index][:i+1]
                                        model[h_index].fit(x_train, np.ravel(y_train))
                                        break

                            #save last trained model, and add to training summary
                            training_summary = training_summary.append(summarize(self, model, error_list, relative_error_list, hit_list, 
                                                                                params, total_train_time, total_test_time))

                        completed_steps += 1
                        update_progress(100.0 * completed_steps/total_steps)
        return training_summary, make_presentable(training_summary, self.summary_name)

    def backtest(self, (sample_presentation, use_full_indicators, n_estimators, max_features, min_samples_split), h_index, df, initialization_samples = 1250):
        """ Train and validate regressor on all samples in dataframe. """
        if use_full_indicators:
            first_sample, samples, labels = prepare_samples(df, self.full_indicators_samples)
            daily_samples = self.full_indicators_samples['Daily']
        else:
            first_sample, samples, labels = prepare_samples(df, self.indicators_samples)
            daily_samples = self.indicators_samples['Daily']

        presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, daily_samples)
        model, y_predict = [[0 for i in range (len(h))] for j in range(2)]
        return_y = []

        for i in range(initialization_samples - max(h) + 1, df.shape[0]): #first 5 years is bootstrapping samples only
            i -= first_sample
            if i>=len(presented_labels[h_index]):
                y_predict = None
            else:
                x_train, x_validate = presented_samples[:i-h[h_index]+1,:], presented_samples[i,:] #need to stop training h steps before test
                y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                #train
                model = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators = n_estimators, 
                                                                                            max_features = max_features, min_samples_split = min_samples_split))
                model.fit(x_train, np.ravel(y_train))
                #predict
                y_predict = model.predict(x_validate.reshape(1,-1))[0]
                y_predict = remove_presentation(y_predict, normalizer[i], sample_presentation)[0]

            return_y.append([labels[0][i], y_predict])
        colnames = ['actual'] + \
                   [self.summary_name+str((sample_presentation, use_full_indicators, n_estimators, max_features, min_samples_split))+', h='+ str(h[h_index])]
        return_df= pd.DataFrame(return_y, columns = colnames)
        #shift each column forward by h to align timelines
        return_df.ix[:,1] = return_df.ix[:,1].shift(h[h_index]-1) #first column is 'actual' and need not be shifted
        return_df = return_df._ix[max(h)-1:,:]
        return return_df

    def predict(self, (sample_presentation, use_full_indicators, n_estimators, max_features, min_samples_split), h_index, df):
        """ Forecast beyond last sample in the dataframe using pretrained model. """
        if use_full_indicators:
            sample = prepare_last_sample(df, self.full_indicators_samples)
            daily_samples = self.full_indicators_samples['Daily']
        else:
            sample = prepare_last_sample(df, self.indicators_samples)
            daily_samples = self.indicators_samples['Daily']
        
        dummy_labels = [[0]]*len(h)
        presented_sample, _, normalizer = set_presentation(sample, dummy_labels, sample_presentation, daily_samples)

        prediction = self.pretrained_model[h_index].predict(presented_sample.reshape(1,-1))[0]
        prediction = remove_presentation(prediction, normalizer[-1], sample_presentation)
        return prediction


class AdaBoost(PredictiveModel):
    """ AdaBoost regression predictive model."""

    def __init__(self):
        """ Initialize predictive model with model, model indicators, and params. """
        self.name = "AdaBoost"
        self.summary_name = "AB"
        self.indicators_samples      = {'Daily':42}
        self.full_indicators_samples = {'Daily':42, 'Volume':10, 'Open':10, 'High':10, 'Low':10, 'SMA':5, 'EWMA':5, 'MOM':5, 'STD':5}
        self.model_params = dict(n_estimators   = [5,25],
                                 learning_rate  = [0.01,0.1,1],
                                 loss           = ['linear', 'exponential'],
                                 full_indicators     = [True, False],
                                 sample_presentation = [SamplePresentation.cumulative])
        self.pretrained_model = None #save the pretrained model for future use

    def train_validate(self, df, validation_range, update_progress):
        """ Train and validate regressor on df samples with indices listed in validation_range. """
        training_summary = pd.DataFrame()

        # progress bar parameters
        total_steps = len(self.model_params['full_indicators']) * \
                      len(self.model_params['sample_presentation']) * len(self.model_params['n_estimators']) * \
                      len(self.model_params['learning_rate'])
        completed_steps = 0

        # loop over model parameters
        for use_full_indicators in self.model_params['full_indicators']:
            if use_full_indicators:
                first_sample, samples, labels = prepare_samples(df, self.full_indicators_samples)
                daily_samples = self.full_indicators_samples['Daily']
            else:
                first_sample, samples, labels = prepare_samples(df, self.indicators_samples)
                daily_samples = self.indicators_samples['Daily']

            for sample_presentation in self.model_params['sample_presentation']:
                presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, daily_samples)
                for n_estimators in self.model_params['n_estimators']:
                    for learning_rate in self.model_params['learning_rate']:
                        for loss in self.model_params['loss']:
                            model, total_train_time, total_test_time = [[0 for i in range (len(h))] for j in range(3)]
                            error_list, relative_error_list, hit_list = [[[] for i in range (len(h))] for j in range(3)]
                            params = (sample_presentation, use_full_indicators, n_estimators, learning_rate, loss)

                            # model training and validation core
                            for h_index in range(len(h)):
                                for index in validation_range:
                                    i = index-first_sample                        
                                    x_train, x_validate = presented_samples[:i-h[h_index]+1,:], presented_samples[i,:] #need to stop training h steps before test
                                    y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                                    #train
                                    t1 = time.time()
                                    model[h_index] = AdaBoostRegressor(n_estimators = n_estimators, learning_rate = learning_rate, loss = loss)
                                    model[h_index].fit(x_train, np.ravel(y_train))
                                    t2 = time.time()
                                    train_time = (t2-t1)
                                    #test
                                    y_predict = model[h_index].predict(x_validate.reshape(1,-1))
                                    test_time = (time.time()-t2)
                                    #apend new results
                                    y_validate_absolute = remove_presentation(y_validate,normalizer[i], sample_presentation)
                                    y_predict_absolute  = remove_presentation(y_predict ,normalizer[i], sample_presentation)
                                    error_list[h_index] += [y_validate_absolute - y_predict_absolute]
                                    relative_error_list[h_index] += [(y_validate_absolute - y_predict_absolute)/y_validate_absolute]
                                    hit_list[h_index] += [(y_validate-x_validate[-1])*(y_predict-x_validate[-1]) > 0]
            
                                    total_train_time[h_index] += train_time
                                    total_test_time[h_index] += test_time
                                    if i == len(presented_labels[h_index])-1:
                                        #very last training point, include last training oppurtunity
                                        x_train = presented_samples[:i+1,:]
                                        y_train = presented_labels[h_index][:i+1]
                                        model[h_index].fit(x_train, np.ravel(y_train))
                                        break

                            #save last trained model, and add to training summary
                            training_summary = training_summary.append(summarize(self, model, error_list, relative_error_list, hit_list, 
                                                                                params, total_train_time, total_test_time))

                        completed_steps += 1
                        update_progress(100.0 * completed_steps/total_steps)
        return training_summary, make_presentable(training_summary, self.summary_name)

    def backtest(self, (sample_presentation, use_full_indicators, n_estimators, learning_rate, loss), h_index, df, initialization_samples = 1250):
        """ Train and validate regressor on all samples in dataframe. """
        if use_full_indicators:
            first_sample, samples, labels = prepare_samples(df, self.full_indicators_samples)
            daily_samples = self.full_indicators_samples['Daily']
        else:
            first_sample, samples, labels = prepare_samples(df, self.indicators_samples)
            daily_samples = self.indicators_samples['Daily']

        presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, daily_samples)
        model, y_predict = [[0 for i in range (len(h))] for j in range(2)]
        return_y = []

        for i in range(initialization_samples - max(h) + 1, df.shape[0]): #first 5 years is bootstrapping samples only
            i -= first_sample
            if i>=len(presented_labels[h_index]):
                y_predict = None
            else:
                x_train, x_validate = presented_samples[:i-h[h_index]+1,:], presented_samples[i,:] #need to stop training h steps before test
                y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                #train
                model = AdaBoostRegressor(n_estimators = n_estimators, learning_rate = learning_rate, loss = loss)
                model.fit(x_train, np.ravel(y_train))
                #predict
                y_predict = model.predict(x_validate.reshape(1,-1))[0]
                y_predict = remove_presentation(y_predict, normalizer[i], sample_presentation)[0]

            return_y.append([labels[0][i], y_predict])
        colnames = ['actual'] + \
                   [self.summary_name+str((sample_presentation, use_full_indicators, n_estimators, learning_rate, loss))+', h='+ str(h[h_index])]
        return_df= pd.DataFrame(return_y, columns = colnames)
        #shift each column forward by h to align timelines
        return_df.ix[:,1] = return_df.ix[:,1].shift(h[h_index]-1) #first column is 'actual' and need not be shifted
        return_df = return_df._ix[max(h)-1:,:]
        return return_df

    def predict(self, (sample_presentation, use_full_indicators, n_estimators, learning_rate, loss), h_index, df):
        """ Forecast beyond last sample in the dataframe using pretrained model. """
        if use_full_indicators:
            sample = prepare_last_sample(df, self.full_indicators_samples)
            daily_samples = self.full_indicators_samples['Daily']
        else:
            sample = prepare_last_sample(df, self.indicators_samples)
            daily_samples = self.indicators_samples['Daily']
        
        dummy_labels = [[0]]*len(h)
        presented_sample, _, normalizer = set_presentation(sample, dummy_labels, sample_presentation, daily_samples)

        prediction = self.pretrained_model[h_index].predict(presented_sample.reshape(1,-1))[0]
        prediction = remove_presentation(prediction, normalizer[-1], sample_presentation)
        return prediction


class GradientBoost(PredictiveModel):
    """ AdaBoost regression predictive model."""

    def __init__(self):
        """ Initialize predictive model with model, model indicators, and params. """
        self.name = "Gradient Boosting"
        self.summary_name = "GB"
        self.indicators_samples      = {'Daily':42}
        self.full_indicators_samples = {'Daily':42, 'Volume':10, 'Open':10, 'High':10, 'Low':10, 'SMA':5, 'EWMA':5, 'MOM':5, 'STD':5}
        self.model_params = dict(n_estimators   = [5,25,100],
                                 learning_rate  = [0.01,0.1],
                                 loss           = ['lad', 'huber'],
                                 full_indicators     = [True, False],
                                 sample_presentation = [SamplePresentation.cumulative])
        self.pretrained_model = None #save the pretrained model for future use

    def train_validate(self, df, validation_range, update_progress):
        """ Train and validate regressor on df samples with indices listed in validation_range. """
        training_summary = pd.DataFrame()

        # progress bar parameters
        total_steps = len(self.model_params['full_indicators']) * \
                      len(self.model_params['sample_presentation']) * len(self.model_params['n_estimators']) * \
                      len(self.model_params['learning_rate'])
        completed_steps = 0

        # loop over model parameters
        for use_full_indicators in self.model_params['full_indicators']:
            if use_full_indicators:
                first_sample, samples, labels = prepare_samples(df, self.full_indicators_samples)
                daily_samples = self.full_indicators_samples['Daily']
            else:
                first_sample, samples, labels = prepare_samples(df, self.indicators_samples)
                daily_samples = self.indicators_samples['Daily']

            for sample_presentation in self.model_params['sample_presentation']:
                presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, daily_samples)
                for n_estimators in self.model_params['n_estimators']:
                    for learning_rate in self.model_params['learning_rate']:
                        for loss in self.model_params['loss']:
                            model, total_train_time, total_test_time = [[0 for i in range (len(h))] for j in range(3)]
                            error_list, relative_error_list, hit_list = [[[] for i in range (len(h))] for j in range(3)]
                            params = (sample_presentation, use_full_indicators, n_estimators, learning_rate, loss)

                            # model training and validation core
                            for h_index in range(len(h)):
                                for index in validation_range:
                                    i = index-first_sample                        
                                    x_train, x_validate = presented_samples[:i-h[h_index]+1,:], presented_samples[i,:] #need to stop training h steps before test
                                    y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                                    #train
                                    t1 = time.time()
                                    model[h_index] = GradientBoostingRegressor(n_estimators = n_estimators, learning_rate = learning_rate, loss = loss)
                                    model[h_index].fit(x_train, np.ravel(y_train))
                                    t2 = time.time()
                                    train_time = (t2-t1)
                                    #test
                                    y_predict = model[h_index].predict(x_validate.reshape(1,-1))
                                    test_time = (time.time()-t2)
                                    #apend new results
                                    y_validate_absolute = remove_presentation(y_validate,normalizer[i], sample_presentation)
                                    y_predict_absolute  = remove_presentation(y_predict ,normalizer[i], sample_presentation)
                                    error_list[h_index] += [y_validate_absolute - y_predict_absolute]
                                    relative_error_list[h_index] += [(y_validate_absolute - y_predict_absolute)/y_validate_absolute]
                                    hit_list[h_index] += [(y_validate-x_validate[-1])*(y_predict-x_validate[-1]) > 0]
            
                                    total_train_time[h_index] += train_time
                                    total_test_time[h_index] += test_time
                                    if i == len(presented_labels[h_index])-1:
                                        #very last training point, include last training oppurtunity
                                        x_train = presented_samples[:i+1,:]
                                        y_train = presented_labels[h_index][:i+1]
                                        model[h_index].fit(x_train, np.ravel(y_train))
                                        break

                            #save last trained model, and add to training summary
                            training_summary = training_summary.append(summarize(self, model, error_list, relative_error_list, hit_list, 
                                                                                params, total_train_time, total_test_time))

                        completed_steps += 1
                        update_progress(100.0 * completed_steps/total_steps)
        return training_summary, make_presentable(training_summary, self.summary_name)

    def backtest(self, (sample_presentation, use_full_indicators, n_estimators, learning_rate, loss), h_index, df, initialization_samples = 1250):
        """ Train and validate regressor on all samples in dataframe. """
        if use_full_indicators:
            first_sample, samples, labels = prepare_samples(df, self.full_indicators_samples)
            daily_samples = self.full_indicators_samples['Daily']
        else:
            first_sample, samples, labels = prepare_samples(df, self.indicators_samples)
            daily_samples = self.indicators_samples['Daily']

        presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, daily_samples)
        model, y_predict = [[0 for i in range (len(h))] for j in range(2)]
        return_y = []

        for i in range(initialization_samples - max(h) + 1, df.shape[0]): #first 5 years is bootstrapping samples only
            i -= first_sample
            if i>=len(presented_labels[h_index]):
                y_predict = None
            else:
                x_train, x_validate = presented_samples[:i-h[h_index]+1,:], presented_samples[i,:] #need to stop training h steps before test
                y_train, y_validate = presented_labels[h_index][:i-h[h_index]+1], presented_labels[h_index][i]
                #train
                model = GradientBoostingRegressor(n_estimators = n_estimators, learning_rate = learning_rate, loss = loss)
                model.fit(x_train, np.ravel(y_train))
                #predict
                y_predict = model.predict(x_validate.reshape(1,-1))[0]
                y_predict = remove_presentation(y_predict, normalizer[i], sample_presentation)[0]

            return_y.append([labels[0][i], y_predict])
        colnames = ['actual'] + \
                   [self.summary_name+str((sample_presentation, use_full_indicators, n_estimators, learning_rate, loss))+', h='+ str(h[h_index])]
        return_df= pd.DataFrame(return_y, columns = colnames)
        #shift each column forward by h to align timelines
        return_df.ix[:,1] = return_df.ix[:,1].shift(h[h_index]-1) #first column is 'actual' and need not be shifted
        return_df = return_df._ix[max(h)-1:,:]
        return return_df

    def predict(self, (sample_presentation, use_full_indicators, n_estimators, learning_rate, loss), h_index, df):
        """ Forecast beyond last sample in the dataframe using pretrained model. """
        if use_full_indicators:
            sample = prepare_last_sample(df, self.full_indicators_samples)
            daily_samples = self.full_indicators_samples['Daily']
        else:
            sample = prepare_last_sample(df, self.indicators_samples)
            daily_samples = self.indicators_samples['Daily']
        
        dummy_labels = [[0]]*len(h)
        presented_sample, _, normalizer = set_presentation(sample, dummy_labels, sample_presentation, daily_samples)

        prediction = self.pretrained_model[h_index].predict(presented_sample.reshape(1,-1))[0]
        prediction = remove_presentation(prediction, normalizer[-1], sample_presentation)
        return prediction


class StackedGeneralization(PredictiveModel):
    """ Stacked generalization regression predictive model."""

    def __init__(self):
        """ Initialize predictive model with model, model indicators, and params. """
        self.name = "Stacked Generalization"
        self.summary_name = "SG"
        self.indicators_samples      = {'Daily':42}
        self.full_indicators_samples = {'Daily':42, 'Volume':21, 'SMA':21, 'EWMA':21, 'MOM':21, 'STD':21}
        self.model_params = dict(l1_generalizer   = ['GB'],
                                 l0_generalizers  = [[Knn, LeastSquares]],#[[Knn, LeastSquares, SVReg, RandomForest, AdaBoost, GradientBoost]],
                                 sample_presentation = [SamplePresentation.absolute, SamplePresentation.cumulative])
        self.n_best_models = 5
        #instantiate and list training models
        self.pretrained_model = None #save the pretrained model for future use
        self.submodels_summary = None #save the pretrained submodels for future use

    def train_validate(self, df, validation_range, update_progress):
        """ Train and validate regressor on df samples with indices listed in validation_range. """

        # prepare samples for l1 generalizer
        best_models = [pd.DataFrame()]*len(h)
        samples = [np.array([])]*len(h)
        labels  = [0]*len(h)
        #loop over l0 generalizers
        for l0_generalizers in self.model_params['l0_generalizers']:
            for l0_generalizer in l0_generalizers:
                #for each generalizer: train_validate with last 84 samples (4 months) of first year to avoid bias in validation
                model = l0_generalizer()
                submodels_validation_range = range(252-84,252)
                training_summary, _ = model.train_validate(df, submodels_validation_range, update_progress)
                #loop over n best l0 generalizers
                for h_index in range(len(h)):
                    training_models = pick_best(training_summary, h[h_index], self.n_best_models, True)
                    for index, training_model in training_models.iterrows():
                        #backtest generalizer with all samples and get result
                        backtest_df = training_model['trained_model'].backtest(training_model['params'], h_index, df, 252) 
                        #add results from backtest to training
                        labels[h_index] = backtest_df.ix[:,-2].tolist()
                        samples[h_index] = np.append(samples[h_index],backtest_df.ix[:,-1].tolist())
                    #training is complete ...
                    best_models[h_index] = pd.concat([best_models[h_index], training_models])
        self.submodels_summary = best_models

        samples = [np.reshape(samples[h_index],(-1, len(labels[h_index]))).T for h_index in range(len(h))]
        # progress bar parameters
        total_steps = len(self.model_params['sample_presentation']) * \
                      len(self.model_params['l1_generalizer'])
        completed_steps = 0

        training_summary = pd.DataFrame()
        #loop over l1 generalizers
        for l1_generalizer in self.model_params['l1_generalizer']:
            # model training and validation core
            for sample_presentation in self.model_params['sample_presentation']:
                params = (l1_generalizer, sample_presentation)
                model, total_train_time, total_test_time = [[0 for i in range (len(h))] for j in range(3)]
                error_list, relative_error_list, hit_list = [[[] for i in range (len(h))] for j in range(3)]
                for h_index in range(len(h)):
                    samples_shift = df.shape[0]-samples[h_index].shape[0]
                    presented_samples, presented_labels, normalizer = set_presentation(samples[h_index], labels, sample_presentation, samples[h_index].shape[1])
                    for index in validation_range:
                        i = index-samples_shift
                        if i == 0:
                            continue
                        x_train, x_validate = presented_samples[:i+1-h[h_index],:], presented_samples[i,:] #l1 generalizer can stop training on last l0 sample
                        y_train, y_validate = presented_labels[h_index][:i+1-h[h_index]], presented_labels[h_index][i]
            
                        #train and validate l1 generalizer
                        t1 = time.time()
                        if l1_generalizer == 'LR':                            
                            model[h_index] = Ridge()
                        elif l1_generalizer == 'SVR':
                            model[h_index] = make_pipeline(preprocessing.StandardScaler(), svm.SVR())
                        elif l1_generalizer == 'RF': 
                            model[h_index] = RandomForestRegressor()
                        elif l1_generalizer == 'GB':
                            model[h_index] = GradientBoostingRegressor()
                        model[h_index].fit(x_train, np.ravel(y_train))
                        t2 = time.time()
                        train_time = (t2-t1)
                        #test
                        y_predict = model[h_index].predict(x_validate.reshape(1,-1))
                        test_time = (time.time()-t2)
                        #apend new results
                        y_validate_absolute = remove_presentation(y_validate,normalizer[i], sample_presentation)
                        y_predict_absolute  = remove_presentation(y_predict ,normalizer[i], sample_presentation)
                        error_list[h_index] += [y_validate_absolute - y_predict_absolute]
                        relative_error_list[h_index] += [(y_validate_absolute - y_predict_absolute)/y_validate_absolute]
                        y_reference_absolute = remove_presentation(presented_labels[h_index][i-h[h_index]] ,normalizer[i-h[h_index]], sample_presentation)
                        hit_list[h_index] += [(y_validate_absolute-y_reference_absolute)*(y_predict_absolute-y_reference_absolute) > 0]
            
                        total_train_time[h_index] += train_time
                        total_test_time[h_index] += test_time
                        if i == len(presented_labels[h_index])-1:
                            #very last training point, include last training oppurtunity
                            x_train = presented_samples[:i+1,:]
                            y_train = presented_labels[h_index][:i+1]
                            model[h_index].fit(x_train, np.ravel(y_train))
                            break

                # save last trained model, and add to training summary
                training_summary = training_summary.append(summarize(self, model, error_list, relative_error_list, hit_list, 
                                                                    params, total_train_time, total_test_time))

                completed_steps += 1
                update_progress(100.0 * completed_steps/total_steps)
        return training_summary, make_presentable(training_summary, self.summary_name)

    def backtest(self, (l1_generalizer, sample_presentation), h_index, df, initialization_samples = 1250):
        """ Train and validate regressor on all samples in dataframe. """
        #build the samples using l0 predictions
        samples = [np.array([])]
        labels  = [0]*len(h)
        for _, training_model in self.submodels_summary[h_index].iterrows():
            #backtest generalizer with all samples and get result
            backtest_df = training_model['trained_model'].backtest(training_model['params'], h_index, df) 
            #add results from backtest to training
            samples = np.append(samples,backtest_df.ix[:,-1].tolist())
        
        labels = [backtest_df.ix[:,-2].tolist() for index in range(len(h))]
        samples = np.reshape(samples,(-1, len(labels[h_index]))).T

        #use the sample in l1 prediction
        presented_samples, presented_labels, normalizer = set_presentation(samples, labels, sample_presentation, samples.shape[1])
        model, y_predict = [0],[0]
        return_y = []
        for i in range(samples.shape[0]):
            if i<=h[h_index]:
                y_predict = None
            else:
                x_train, x_validate = presented_samples[:i+1-h[h_index],:], presented_samples[i,:] #l1 generalizer can stop training on last l0 sample
                y_train, y_validate = presented_labels[h_index][:i+1-h[h_index]], presented_labels[h_index][i]
            
                #train and validate l1 generalizer
                if l1_generalizer == 'LR':                            
                    model = Ridge()
                elif l1_generalizer == 'SVR':
                    model = make_pipeline(preprocessing.StandardScaler(), svm.SVR())
                elif l1_generalizer == 'RF': 
                    model = RandomForestRegressor()
                elif l1_generalizer == 'GB':
                    model = GradientBoostingRegressor()
                model.fit(x_train, np.ravel(y_train))
                #predict
                y_predict = model.predict(x_validate.reshape(1,-1))[0]
                y_predict = remove_presentation(y_predict, normalizer[i], sample_presentation)
                if type(y_predict) == type(np.array([])):
                    y_predict = y_predict[0]
            
            return_y.append([labels[0][i], y_predict])
        colnames = ['actual'] + \
                   [self.summary_name+str((l1_generalizer, sample_presentation))+', h='+ str(h[h_index])]
        return_df= pd.DataFrame(return_y, columns = colnames)
        return return_df

    def predict(self, (l1_generalizer, sample_presentation), h_index, df):
        """ Forecast beyond last sample in the dataframe using pretrained model. """
        #build the sample using l0 predictions
        sample = np.array([])
        for _, training_model in self.submodels_summary[h_index].iterrows():
            sample = np.append(sample, training_model['trained_model'].predict(training_model['params'], h_index, df))

        #use the sample in l1 prediction
        dummy_labels = [[0]]*len(h)
        sample = np.reshape(sample,(1,-1))
        presented_sample, _, normalizer = set_presentation(sample, dummy_labels, sample_presentation, sample.shape[1])
        prediction = self.pretrained_model[h_index].predict(presented_sample)[0]
        prediction = remove_presentation(prediction, normalizer[-1], sample_presentation)
        return prediction