""" This module holds the initial GUI of the app and connects to different controllers
and views in the project.
"""


#.-------------------.
#|      imports      |
#'-------------------'
import matplotlib
matplotlib.use('TkAgg')

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import ttk

from st_explore_controller import *
from st_train_controller import *
from st_backtest_controller import *
from st_forecast_controller import *
from st_helpers import StColors

def set_app_style():
    style = ttk.Style()
    style.theme_create( "st_app", parent="alt", settings={
        ".":             {"configure": {"background"      : StColors.dark_grey,
                                        "foreground"      : StColors.light_grey,
                                        "relief"          : "flat",
                                        "highlightcolor"  : StColors.bright_green}},

        "TLabel":        {"configure": {"foreground"      : StColors.bright_green,
                                        "padding"         : 10,
                                        "font"            : ("Calibri", 12)}},

        "TNotebook":     {"configure": {"padding"         : 5}},
        "TNotebook.Tab": {"configure": {"padding"         : [25, 5], 
                                        "foreground"      : "white"},
                            "map"      : {"background"      : [("selected", StColors.mid_grey)],
                                        "expand"          : [("selected", [1, 1, 1, 0])]}},

        "TCombobox":     {"configure": {"selectbackground": StColors.dark_grey,
                                        "fieldbackground" : "white",
                                        "background"      : StColors.light_grey,
                                        "foreground"      : "black"}},

        "TButton":       {"configure": {"font"            :("Calibri", 13, 'bold'),
                                        "background"      : "black",
                                        "foreground"      : StColors.bright_green},
                            "map"      : {"background"      : [("active", StColors.bright_green)],
                                        "foreground"      : [("active", 'black')]}},
            
        "TEntry":        {"configure": {"foreground"      : "black"}},
        "Horizontal.TProgressbar":{"configure": {"background": StColors.mid_grey}}
    })
    style.theme_use("st_app")

#.----------------.
#|    classes     |
#'----------------'
class MainView(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        tk.Tk.wm_title(self, "Smart Trader System")
        if os.name == "nt":
            tk.Tk.iconbitmap(self, default="app_icon.ico")
        self.configure(background = StColors.dark_grey)
        set_app_style()

        # Top section (header)
        title = ttk.Label(self, text="Smart Trader Portal", font=("Tahoma", 25, 'bold'))
        title.pack()
        
        # Lower section (tabs area)
        notebook = ttk.Notebook(self)
        notebook.pack()
        # Create Controllers and add their views
        explore = ExploreController(self)
        train   = TrainingController(self)
        backtest= BacktestController(self)
        forecast= ForecastController(self)
        for view_frame in (explore.view, train.view, backtest.view, forecast.view):
            view_frame.grid(row=0, column=0, sticky="nsew")
            notebook.add(view_frame, text = view_frame.title)