"""View file for backtesting
"""

#.-------------------.
#|      imports      |
#'-------------------'
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
class BacktestView(ttk.Frame):
    """ View class for backtesting
	   
    The view class handles GUI and communicates user requests to controller following
    the MVC design pattern.

    Args:
        - parent_view (tk widget): the view object which this backtest view will
        be part of.
        - controller (BacktestController): controller which will handle user requests.

	"""
    def __init__(self, parent, controller):
        self.controller = controller        
        self.title = "Backtest"
        ttk.Frame.__init__(self, parent)

        # Left Pane
        left_frame = ttk.Frame(self, padding = 20)
        left_frame.pack(side = "left")

        label = ttk.Label(left_frame, text= "Pre-trained Stock")
        label.pack(padx=10, anchor = tk.W)
        symbol_list = ttk.Combobox(left_frame, width = 18, font=("Calibri", 12))
        symbol_list.pack(padx = 20, pady = (0,35), anchor = tk.W)

        label = ttk.Label(left_frame, text= "Start Date - ex: 2015-03-31")
        label.pack(padx=10, anchor = tk.W, )
        start_entry = ttk.Entry(left_frame, width = 20, font=("Calibri", 12))
        start_entry.pack(padx = 20, anchor = tk.W)

        label = ttk.Label(left_frame, text= "End Date")
        label.pack(padx=10, anchor = tk.W, )
        end_entry = ttk.Entry(left_frame, width = 20, font=("Calibri", 12))
        end_entry.pack(padx = 20, pady = (0,35), anchor = tk.W)

        label = ttk.Label(left_frame, text= "Predictive Model")
        label.pack(padx=10, anchor = tk.W, )
        models = self.controller.get_trained_models(symbol_list.get())
        type_list = ttk.Combobox(left_frame, width = 18, font=("Calibri", 12))
        type_list.pack(padx = 20, anchor = tk.W)

        run_btn = ttk.Button(left_frame, text= "Run Backtesting", padding = (25,10,25,10),
                             command=lambda: run_backtest())
        run_btn.pack(pady=50,padx=15, anchor = tk.W)

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
        self.bind("<Visibility>", update_selections)

        def symbol_change_handler(event):
            trained_models = self.controller.get_trained_models(symbol_list.get())
            type_list['value'] = trained_models
            if len(trained_models) == 0:
                type_list['value'] = ["No trained models found"]
            type_list.current(0)
        symbol_list.bind("<<ComboboxSelected>>", symbol_change_handler)

        def on_key_event(event):
            key_press_handler(event, canvas, toolbar)
        canvas.mpl_connect('key_press_event', on_key_event)

        def on_click(event):
            canvas._tkcanvas.focus_set()
        canvas.mpl_connect('button_press_event', on_click)

        def run_backtest():
            # extract variables needed for execuation
            symbol = symbol_list.get()
            start_date = start_entry.get()
            end_date = end_entry.get()
            model = type_list.get()
        
		    # run selected model backtest
            try:
                plot = self.controller.run_backtest(symbol, model, start_date, end_date)
            except ValidationError as input_error:
                print input_error.message

            figure.clear()
            plot_data(figure, plot)
            canvas.draw()