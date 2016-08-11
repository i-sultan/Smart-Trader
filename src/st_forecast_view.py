"""View file for forecasting
"""

#.-------------------.
#|      imports      |
#'-------------------'
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import ttk
from st_helpers import *

#.----------------.
#|    classes     |
#'----------------'
class ForecastView(ttk.Frame):
    """ View class for forecasting

  	The view class handles GUI and communicates user requests to controller following
    the MVC design pattern.

    Args:
        - parent_view (tk widget): the view object which this backtest view will
        be part of.
        - controller (ForecastController): controller which will handle user requests.

	"""
    def __init__(self, parent, controller):
        self.controller = controller        
        self.title = "Forecast"
        ttk.Frame.__init__(self, parent)

        # Left Pane
        left_frame = ttk.Frame(self, padding = 20)
        left_frame.pack(side = "left")

        label = ttk.Label(left_frame, text= "Pre-trained Stock")
        label.pack(padx=10, anchor = tk.W)
        symbol_list = ttk.Combobox(left_frame, width = 18, font=("Calibri", 12))
        symbol_list.pack(padx = 20, anchor = tk.W)

        label = ttk.Label(left_frame, text= "Predictive Model")
        label.pack(padx=10, anchor = tk.W, )
        type_list = ttk.Combobox(left_frame, width = 18, font=("Calibri", 12))
        type_list.pack(padx = 20, pady = (0,25), anchor = tk.W)

        label = ttk.Label(left_frame, text= "Training Start Date")
        label.pack(padx=10, anchor = tk.W)
        start_var = tk.StringVar()
        start_label = ttk.Label(left_frame, textvariable = start_var, foreground = StColors.light_grey)
        start_label.pack(padx = 30, pady = 0, anchor = tk.W)

        label = ttk.Label(left_frame, text= "Training End Date")
        label.pack(padx=10, anchor = tk.W, )
        end_var = tk.StringVar()
        end_label = ttk.Label(left_frame, textvariable = end_var, foreground = StColors.light_grey)
        end_label.pack(padx = 30, anchor = tk.W)

        label = ttk.Label(left_frame, text= "Prediction Date")
        label.pack(padx=10, anchor = tk.W, )
        date_entry = ttk.Entry(left_frame, width = 20, font=("Calibri", 12))
        date_entry.pack(padx = 20, anchor = tk.W)

        run_btn = ttk.Button(left_frame, text= "Forecast Plot", padding = (30,5,30,5),
                             command=lambda: run_forecast_plot())
        run_btn.pack(pady=(50,0),padx=15, anchor = tk.W)

        run_btn = ttk.Button(left_frame, text= "Forecast Table", padding = (25,5,25,5),
                             command=lambda: run_forecast_table())
        run_btn.pack(pady=(5,50),padx=15, anchor = tk.W)

        # Right Pane
        right_frame = ttk.Frame(self, padding = 20)
        right_frame.pack(side = "right")

        figure = Figure(figsize=(12, 8), dpi=60, facecolor = 'white')
        canvas = FigureCanvasTkAgg(figure, right_frame)
        canvas.get_tk_widget().grid(row=0,column=1)
        toolbar = NavigationToolbar2TkAgg(canvas, right_frame)
        toolbar.grid(row=1,column=1, pady = 20)

        def update_selections(event):
            trained_symbols = self.controller.get_trained_symbols()
            symbol_list['value']= trained_symbols
            if trained_symbols:
                symbol_list.current(0)
            models = self.controller.get_trained_models(symbol_list.get())
            type_list['value']= models
            if trained_symbols:
                type_list.current(0)
            trained_start_date, trained_end_date = self.controller.get_trained_dates(symbol_list.get(), type_list.get())
            start_var.set(trained_start_date)
            end_var.set(trained_end_date)
        self.bind("<Visibility>", update_selections)

        def symbol_change_handler(event):
            # update models list
            trained_models = self.controller.get_trained_models(symbol_list.get())
            type_list['value'] = trained_models
            if len(trained_models) == 0:
                type_list['value'] = ["No trained models found"]
                start_label.config(text="No trained models")
                end_label.config(text="No trained models")
                type_list.current(0)
            else:
                type_list.current(0)
                # update dates
                trained_start_date, trained_end_date = self.controller.get_trained_dates(symbol_list.get(), type_list.get())
                start_label.config(text=trained_start_date)
                end_label.config(text=trained_start_date)
        symbol_list.bind("<<ComboboxSelected>>", symbol_change_handler)

        def model_change_handler(event):
            trained_models = self.controller.get_trained_models(symbol_list.get())
            if len(trained_models) != 0:
                # update dates
                trained_start_date, trained_end_date = self.controller.get_trained_dates(symbol_list.get(), type_list.get())
                start_label.config(text=trained_start_date)
                end_label.config(text=trained_end_date)
        type_list.bind("<<ComboboxSelected>>", model_change_handler)

        def on_key_event(event):
            key_press_handler(event, canvas, toolbar)
        canvas.mpl_connect('key_press_event', on_key_event)

        def on_click(event):
            canvas._tkcanvas.focus_set()
        canvas.mpl_connect('button_press_event', on_click)

        def run_forecast_plot():
            # extract variables needed for execuation
            symbol = symbol_list.get()
            model = type_list.get()
            prediction_date = date_entry.get()
        
		    # run selected model backtest
            try:
                plot, table = self.controller.run_forecast(symbol, model, prediction_date)
            except ValidationError as input_error:
                print input_error.message

            figure.clear()
            plot_data(figure, plot)
            canvas.draw()

        def run_forecast_table():
            # extract variables needed for execuation
            symbol = symbol_list.get()
            model = type_list.get()
            prediction_date = date_entry.get()
        
		    # run selected model backtest
            try:
                plot, table = self.controller.run_forecast(symbol, model, prediction_date)
            except ValidationError as input_error:
                print input_error.message

            figure.clear()
            tabulate_data(figure, table)
            canvas.draw()
            canvas.draw()