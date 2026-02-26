"""
Neuro-Evolution Driver Example
~~
DESCRIPTION OF PROBLEM
classic polecart problem
starts a pole randomly in bottom third of circle to see if it can swing it up
"""
# -------- IMPORTS --------
import sys
sys.path.append("../..") # point to relative location of /eco_6
import eco_6.ecosys as eco
import torch
from eco_6.timing import timing
import random
import math

"""
# allow half precision matmul speedup
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
"""

# -------- DIRECTOR --------
ndir = eco.evo.NevoDirector(
    torch.device(type="cuda"),
    torch.float32,
    [ # custom config overrides
        "mainGrid",
        # "debug",
        "production",
    ]
)
# getting some stats back from grid
numTimesteps = ndir.masterConfig["sim"]["numTimesteps"]
numGenerations = ndir.masterConfig["sim"]["numGenerations"]
popSize = ndir.masterConfig["sim"]["popSize"]


# -------- SESSION --------
class Session(eco.esu.SessionUtils):
    """
    Session contains everything involved in setting up, testing, logging, and scoring the problem at hand
    --
    utils: 
    p() easy class printing
    various graphing methods
    time tracking
    --
    Class self.Tensors [size]: additional info
    """
    def __init__(self):
        eco.esu.SessionUtils.__init__(self, graph=ndir.masterConfig["sim"]["graphing"]) # for time tracking
        self.timelineA = torch.tensor([ 1, .4, .2, 0, .8], **ndir.gconf) # ans:  1
        self.timelineB = torch.tensor([.2, .4, .2, 0, .8], **ndir.gconf) # ans: .2
        self.guessA = torch.zeros([popSize], **ndir.gconf)
        self.guessB = torch.zeros([popSize], **ndir.gconf)
        self.score = torch.zeros([popSize], **ndir.gconf)
        
    @timing
    def trainTestA(self, tsIndex: int):
        """
        Reality Testing
        Access to: ndir.getRequiredFeatureShape(), ndir.feedForward(featureSet, inference)
        featureSet size must be either:
        3d: [popSize, 1, featureInputLength]
        1d (broadcast): [featureInputLength]
        """
        res = ndir.feedForward(self.timelineA[tsIndex].view([1])) # [popSize, 1, 1]
        self.guessA = res
    
    @timing
    def trainTestB(self, tsIndex: int):
        res = ndir.feedForward(self.timelineB[tsIndex].view([1])) # [popSize, 1, 1]
        self.guessB = res
        
    @timing
    def worldScore(self):
        """
        Reality Testing
        Access to: ndir.getRequiredFeatureShape(), ndir.feedForward(featureSet, inference)
        featureSet size must be either:
        3d: [popSize, 1, featureInputLength]
        1d (broadcast): [featureInputLength]
        """
        
        # abs diff
        self.guessA = torch.abs(self.guessA - 1)
        self.guessB = torch.abs(self.guessB - .2)
        # self.p("self.guessA")
        # self.p("self.guessB")
        
        self.score = (1 - self.guessA - self.guessB).view([-1])
        # self.p("self.score2")
    
    
ssn = Session()
#ssn.graph.ax.set_ylim([-.2, 1])

# print(f"{ndir.grid.textureCrate['act_lstm_long_mem']}")
# ndir.grid.resetMemory()
# ndir.grid.textureCrateContents()

# -------- LOOP --------
for gen in range(numGenerations):
    ssn.printStartLoop(gen)
    
    # -------- TEST & SCORE --------
    ndir.grid.resetMemory()
    for ts in range(numTimesteps): ssn.trainTestA(ts)
    ndir.grid.resetMemory()
    for ts in range(numTimesteps): ssn.trainTestB(ts)
    ssn.worldScore()

    # -------- TEMPERATURE CONTROL --------
    ssn.interruptGPU()

    # -------- EVOLVE --------
    ndir.evoStep(ssn.score) # evolution
    tGraph = ndir.getPerfGraphSlice(ssn.score) # grab highest and lowest elite to graph
    print(f"{tGraph[0]} << best -- ", end="")
    ssn.timeTrackUpdate(ndir.getEvoTimeTracking()) # time tracking, this gets session stats too

ndir.grid.textureCrateContents()
"""
    # -------- LOGIC TICK --------
    if gen % 10 == 0:
        ndir.exportGrid() # savestate
    
    # -------- GRAPH UPDATE, EXPORT, TIMING, & FREEZE --------
    ssn.updateGraphTensor(tGraph) # update internal tensor only
    if gen % 10 == 0: ssn.redrawGraph() # expensive redraw every %x logic ticks
    # ssn.redrawGraph() # DEBUG

# -------- TIMING STATS --------
ssn.timeTrackOutput()

ssn.redrawGraph()
ssn.freezeGraph()
"""