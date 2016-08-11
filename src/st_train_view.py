"""View file for training
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
class TrainingView(ttk.Frame):
    """ View class for training/backtesting 
	
    The view class handles GUI and communicates user requests to controller following
    the MVC design pattern.

    Args:
        - parent_view (tk widget): the view object which this backtest view will
        be part of.
        - controller (TrainingController): controller which will handle user requests.

	"""
    def __init__(self, parent, controller):
        self.controller = controller        
        self.title = "Train"
        ttk.Frame.__init__(self, parent)

        # Left Pane
        left_frame = ttk.Frame(self, padding = 20)
        left_frame.pack(side = "left")

        label = ttk.Label(left_frame, text= "Stock(s) - ex: AAPL, GOOG")
        label.pack(padx=10, anchor = tk.W)
        symbol_entry = ttk.Entry(left_frame, width = 20, font=("Calibri", 12)) #entry font cannot be specified with a style
        symbol_entry.pack(padx = 20, pady = (0,35), anchor = tk.W)

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
        training_models = self.controller.get_training_models()
        model_list = ttk.Combobox(left_frame, width = 18, font=("Calibri", 12))
        model_list['value']= training_models
        model_list.current(0)
        model_list.pack(padx = 20, anchor = tk.W)

        run_btn = ttk.Button(left_frame, text= "Start Training", padding = (25,10,25,10),
                             command=lambda: start_training())
        run_btn.pack(pady=50,padx=15, anchor = tk.W)

        # Right Pane
        right_frame = ttk.Frame(self, padding = 20)
        right_frame.pack(side = "right")

        label = ttk.Label(right_frame, text= "Training Progress")
        label.pack(padx = (130,200), anchor = tk.W)

        progress = tk.IntVar(self)
        progress_bar = ttk.Progressbar(right_frame, orient="horizontal", variable = progress, length=400, mode="determinate")
        progress_bar.pack(padx = (10,200), anchor = tk.W)

        file_text = tk.StringVar()
        file_label = ttk.Label(right_frame, textvariable = file_text)
        file_label.pack(padx = (0,200), anchor = tk.W)

        def update_progress(progress_percentage):
            progress.set(progress_percentage)
            progress_bar.update()

        def start_training():
            update_progress(0)
            file_text.set("")
            # Extract variables needed for execuation
            symbols = symbol_entry.get()
            symbols = [symbol.strip() for symbol in symbols.split(',')]
            symbol = [symbols[0]] # if multiple symbols provided, only train over first symbol
            start_date = start_entry.get()
            end_date = end_entry.get()
            training_id = model_list.current()
        
		    # run selected training model
            try:
                training_summary_file = self.controller.run_training(training_id, symbol, start_date, end_date, update_progress)
            except ValidationError as input_error:
                print input_error.message

            file_text.set("Training complete, click here to open training summary")
            file_label.bind("<1>", lambda event: os.system('"' + training_summary_file + '"'))