import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from qbstyles import mpl_style # pip install qbstyles


"""
Simple Multi Line Graph
works with pytorch tensors if converted to cpu first with torch.Tensor.cpu()

------------------------ STATIC GRAPH ------------------------

from graph import MultiLineGraph
graph = MultiLineGraph(
    x_axis_data=[1, 2, 3, 4],
    y_axis_data_arr=[
        [2, 4, 8, 24],
        [3, 5, 2, 1]
    ],
    legend=[
        None,
        "B"
    ],
    y_label="y",
    x_label="x",
    graph_title="Graph"
)
graph.freeze_window()

------------------------ UPDATING / PUSHABLE GRAPH ------------------------

from graph import MultiLineGraph
graph = MultiLineGraph(
    x_axis_data=[],
    y_axis_data_arr=[
        [],
        []
    ],
    legend=[
        "highest elite",
        "lowest elite",
    ],
    y_label="Score",
    x_label="Generation",
    graph_title="Performance over time"
)

# iterative push
for i in range(20):
    graph.append(i, [0 - i, 7]) # (x axis item, [y array items])
    graph.redraw()
graph.freeze_window()

# single mass push
self.graph.set(0, [0,1,2,3], [12,7,-4,2])
self.graph.redraw()
self.graph.freeze_window()
"""

class MultiLineGraph:
    """Make a simple line graph, call freeze_window() when finished with updates to keep it open"""
    
    # styling must go here
    mpl_style(dark=True, minor_ticks=False) # minor ticks false helps alleviate double label issue
    
    
    def __init__(self, 
        x_axis_data: list, 
        y_axis_data_arr: list[list], 
        legend: list[str],
        y_label: str = "y",
        x_label: str = "x",
        size: tuple = (14, 8), # size of graph
        graph_title: str = "Title",
        window_title: str = "Graph"
    ):
        plt.ion() # interactive on
        self.fig = plt.figure(num=window_title, figsize=size)
        
        # single plot
        self.ax = self.fig.add_subplot()
        self.x_data = x_axis_data # store original x data
        self.y_data_arr = y_axis_data_arr # store original y data(s)
        self.plot_data = [] # this helps with graph updating live, array of plots

        for index, y_item in enumerate(self.y_data_arr):
            try:
                this_label = legend[index]
                line_item, = self.ax.plot(self.x_data, y_item, label=this_label) # also has linestyle="-", "--", "-.", ":"
            except IndexError:
                print(f"Graph.LineGraph mismatch number of labels")
                line_item, = self.ax.plot(self.x_data, y_item)
            
            # push
            self.plot_data.append(line_item)

        # custom tick markers
        #ax.set_yticks([0, 1, 2])
        #ax.set_xticks([0, 1, 2, 4, 8]) # does include the end even if no line

        # labels
        plt.title(graph_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        # grid
        plt.minorticks_on()
        plt.grid(True, which="minor", color="#777777", linestyle="--")
        plt.grid(True, which="major", color="#ffffff")
        
        # less padding
        self.fig.tight_layout()
        
        plt.legend()
        plt.show()
    
    def update(self, line_index: int = None, x_data: list = None, y_data: list = None):
        """
        For constant updates, use keep_alive if no new data\n
        Update() is for setting the entire graph to something,\n
        use append() if you want to only push progressive update
        """
        
        if x_data is not None:
            self.plot_data[line_index].set_xdata(x_data)
        
        if y_data is not None:
            self.plot_data[line_index].set_ydata(y_data)
    
    def set(self, line_index, x_data, y_data):
        """
        Set one line's worth of data\n
        note that x axis must be set for every line
        """
        self.plot_data[line_index].set_xdata(x_data)
        self.plot_data[line_index].set_ydata(y_data)
    
    def append(self, new_x_pt, all_new_y_pts):
        """
        ex. new_x_pt=7\n
        all_new_y_pts=[5.6, 7] # assuming 2 lines
        """
        
        # add to current data
        self.x_data.append(new_x_pt)
        for index, y_item in enumerate(all_new_y_pts):
            self.y_data_arr[index].append(y_item)
        
        # set x and y
        for index, y_item in enumerate(self.plot_data):
            self.plot_data[index].set_xdata(self.x_data)
            self.plot_data[index].set_ydata(self.y_data_arr[index])
    
    def redraw(self):
        """Redraw and rescale graph"""
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def keep_alive(self):
        """If window isn't frozen, keep_alive will keep window alive in between updates"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def freeze_window(self):
        """Turn interactive mode off and show() will cause it to keep window open"""
        plt.ioff()
        plt.show()
