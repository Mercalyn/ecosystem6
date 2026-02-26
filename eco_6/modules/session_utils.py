"""
Session utils help with organization
available for all problem drivers
"""
from eco_6.graph import MultiLineGraph
from eco_6.eco_print import EcoPrint
import GPUtil
import time
import torch

class SessionUtils:
    def __init__(self, graph: bool = True):
        # setup time tracking
        self.timeTracking = {}
        
        # setup graphing
        self.graphBool = graph
        if self.graphBool:
            self.graph = MultiLineGraph(
                x_axis_data=[],
                y_axis_data_arr=[
                    [],
                    [],
                    [],
                    [],
                    []
                ],
                legend=[
                    "100% pc",
                    "75% pc",
                    "50% pc",
                    "25% pc",
            "0% pc"
                ],
                y_label="Score",
                x_label="Generation",
                graph_title="Performance over time",
                window_title="Ecosystem -- Evolution Performance"
            )
            # graph tensor
            self.graphTensor2d = None
        
        # colored print
        self.e = EcoPrint()
    
    
    # -------- EASY PRINT --------
    def p(self, varName: str, addInfo: str = ""):
        """
        Utility print, addInfo to give additional output print info like tags etc
        keybound to ctrl + shift + o
        ex: self.p("self.answerKey")
        ex. self.p("self.score", "f")
        """
        strName = varName.split(".")[1]
        print(f"~~ {strName} {addInfo} ~~\n{eval(varName)}", end="\n\n")
    
    def printStartLoop(self, stepNum: int):
        """"""
        self.e.loop(f"GEN: {stepNum}")
        self.e.dgrey(" ...")
    
    
    # -------- TIME TRACKING --------
    def timeTrackUpdate(self, timeTakenDict: dict):
        """Update evolution timing function to accumulate time taken"""
        for k, v in timeTakenDict.items():
            try:
                self.timeTracking[k] += v
            except KeyError:
                self.timeTracking[k] = 0
                self.timeTracking[k] += v
    
    def timeTrackOutput(self):
        """Output evolution total time taken"""
        print("\nProcess time tracker: ")
        self.e.lgrey("total secs -- func.__name__\n")
        for k, v in self.timeTracking.items():
            self.e.magenta(f"{v:.3f}")
            self.e.lgrey(" -- ")
            print(f"{k}()")
    
    
    # -------- GRAPHING --------
    def updateGraphTensor(self, graphTensor1d: torch.Tensor):
        """
        Cat a 1d performance tensor to self.graphTensor2d
        inherits type straight from Evolution.grabEliteRangeScores(), likely float32
        """
        if self.graphBool:
            if self.graphTensor2d == None:
                
                # no tensor exists, reshape then set
                graphTensor1d = torch.reshape(graphTensor1d, [1, -1])
                self.graphTensor2d = graphTensor1d
                #print(f"{self.graphTensor2d}")
            
            else:
                # recurring cat, reshape and cat
                graphTensor1d = torch.reshape(graphTensor1d, [1, -1])
                self.graphTensor2d = torch.cat([self.graphTensor2d, graphTensor1d], dim=0)
                #print(f"{self.graphTensor2d}")
    
    def redrawGraph(self):
        """Expensive function: convert whole tensor to graph to cpu and graph"""
        if self.graphBool:
            # x axis 
            cpuTensor = self.graphTensor2d.cpu()
            xSize = cpuTensor.size()[0]
            xAxis = torch.arange(start=0, end=xSize)
            
            # y axes
            self.graph.set(0, xAxis, cpuTensor[:, 0])
            self.graph.set(1, xAxis, cpuTensor[:, 1])
            self.graph.set(2, xAxis, cpuTensor[:, 2])
            self.graph.set(3, xAxis, cpuTensor[:, 3])
            self.graph.set(4, xAxis, cpuTensor[:, 4])
            
            # redraw
            self.graph.redraw()
    
    def freezeGraph(self):
        """Freeze graph at end to keep the window alive"""
        if self.graphBool: self.graph.freeze_window()
    
    def combineTensors(self, aTensor: torch.Tensor, bTensor: torch.Tensor) -> torch.Tensor:
        """
        Give 2 [5] tensors, return 1 [5] tensor
        """
        combined = aTensor
        combined[2:4] = bTensor[0:2]
        combined[4] = 0
        return combined
    
        
    # -------- TEMP THROTTLING --------
    def getGPUTemp(self):
        """Retrieve GPU Temp (not hotspot)"""
        self.gpuTemp = GPUtil.getGPUs()[0].temperature # must be here otherwise will repeat 1st value forever
        self.e.white(" GPU: ")
        
        # color coded
        if self.gpuTemp > 75.0: # 65.0
            self.e.red(f"{self.gpuTemp} C")
        elif self.gpuTemp > 50.0: # 50.0
            self.e.yellow(f"{self.gpuTemp} C")
        else:
            self.e.cyan(f"{self.gpuTemp} C")
        self.e.dgrey(" ...")
    
    def interruptGPU(self):
        """Interrupt GPU if temps get high"""
        self.getGPUTemp()
        
        # 20 sec pause if over 75 C
        if self.gpuTemp > 75.0: # 75.0
            self.e.red(" GPU HOT! Cooling ...")
            time.sleep(20)
            self.getGPUTemp()
        
        # 7 sec pause if over 65
        elif self.gpuTemp > 65.0: # 65.0
            self.e.yellow(" GPU WARM. Cooling ...")
            time.sleep(7)
            self.getGPUTemp()
    
    