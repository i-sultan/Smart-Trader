"""View file for exploration
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
class ExploreView(ttk.Frame):
    """ View class for exploration.
     
	The view class handles GUI and communicates user requests to controller following
    the MVC design pattern.

    Args:
        - parent_view (tk widget): the view object which this backtest view will
        be part of.
        - controller (ExploreController): controller which will handle user requests.

	"""
    def __init__(self, parent, controller):
        self.controller = controller        
        self.title = "Explore"
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

        label = ttk.Label(left_frame, text= "Exploration Type")
        label.pack(padx=10, anchor = tk.W, )
        explorations = self.controller.get_explorations()
        type_list = ttk.Combobox(left_frame, width = 18, font=("Calibri", 12))
        type_list['value']= explorations
        type_list.current(0)
        type_list.pack(padx = 20, anchor = tk.W)

        run_btn = ttk.Button(left_frame, text= "Run Exploration", padding = (25,10,25,10),
                             command=lambda: run_exploration())
        run_btn.pack(pady=50,padx=15, anchor = tk.W)

        # Right Pane
        right_frame = ttk.Frame(self, padding = 20)
        right_frame.pack(side = "right")

        figure = Figure(figsize=(12, 8), dpi=60, facecolor = 'white')
        canvas = FigureCanvasTkAgg(figure, right_frame)
        canvas.get_tk_widget().grid(row=0,column=1)
        toolbar = NavigationToolbar2TkAgg(canvas, right_frame)
        toolbar.grid(row=1,column=1, pady = 20)

        def on_key_event(event):
            if event.key == ' ': #spacebar
                next_id = (type_list.current()+1) % len(explorations)
                type_list.current(next_id)
                run_exploration()
            key_press_handler(event, canvas, toolbar)
        canvas.mpl_connect('key_press_event', on_key_event)

        def on_click(event):
            canvas._tkcanvas.focus_set()
        canvas.mpl_connect('button_press_event', on_click)

        def run_exploration():
            # Extract variables needed for execuation
            symbols = symbol_entry.get()
            symbols = [symbol.strip() for symbol in symbols.split(',')]
            start_date = start_entry.get()
            end_date = end_entry.get()
            exploration_id = type_list.current()
        
		    # run selected exploration
            try:
                plot = self.controller.run_exploration(exploration_id, symbols, start_date, end_date)
            except ValidationError as input_error:
                print input_error.message

            figure.clear()
            if plot.type == PlotTypes.Table:
                tabulate_data(figure, plot) #table
            else:
                plot_data(figure, plot) #plot                
            canvas.draw()
            canvas.draw() #second call is workaround for a table plotting (bug?) in pandas\matplotlib