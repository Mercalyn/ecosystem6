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
        # "load",
        "graph",
        # "maxEvo",
        "drop",
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
    "x_2d":        [numTimesteps, popSize]: cart x position
    "xDot_2d":     [numTimesteps, popSize]: cart x velocity
    "theta_2d":    [numTimesteps, popSize]: pole angle measured 0 at top
    "thetaDot_2d": [numTimesteps, popSize]: polr angular velocity
    
    "force_2d" [numTimesteps, popSize]: force to apply next frame (nn action/output)
    "score_1d" [self.popSize]: accumulated score of cos(theta), so -1.0 to 1.0 for each frame
    """
    def __init__(self):
        eco.esu.SessionUtils.__init__(self, graph=ndir.masterConfig["sim"]["graphing"]) # for time tracking & graph init
        # self.resetSim()
        
    
    @timing
    def resetSim(self):
        # pregen empties
        self.x_2d        = torch.zeros([numTimesteps, popSize], **ndir.gconf)
        self.xDot_2d     = torch.zeros([numTimesteps, popSize], **ndir.gconf)
        self.thetaDot_2d = torch.zeros([numTimesteps, popSize], **ndir.gconf)
        self.force_2d    = torch.zeros([numTimesteps, popSize], **ndir.gconf)
        self.score_1d    = torch.zeros([popSize], **ndir.gconf)
        # print(f"{ndir.grid.textureCrate['0_lstm_short_mem']}")
        # print(f"{ndir.grid.textureCrate['0_lstm_long_mem']}")
        
        # theta starts random near bottom, pi+-1
        self.theta_2d = torch.zeros([numTimesteps, popSize], **ndir.gconf)
        thisVariance = (random.random() - .5) * 2 # -1 to 1
        thisVariance += math.pi # 2.14 to 4.14 theta
        self.theta_2d[0, :] = thisVariance # broadcast 0th tsIndex to this value
        self.theta_2d[0, :] = self.unrotateTheta(self.theta_2d[0, :]) # get valid theta range
        
        """
        # DEBUG::correct slicing and cat
        self.x_2d[0, 0] = 2
        self.xDot_2d[0, 0] = 3
        self.theta_2d[0, 0] = 4
        self.thetaDot_2d[0, 0] = 5
        """
    
        # DEBUG::show init
        # print(f"INIT")
        # self.worldTable()
    
    
    def worldTable(self):
        """Show current table of world physics history"""
        self.p("self.x_2d")
        self.p("self.xDot_2d")
        self.p("self.theta_2d")
        self.p("self.thetaDot_2d")
        self.p("self.force_2d")
        self.p("self.score_1d")
        

    @timing
    def trainTest(self, tsIndex: int):
        """
        Reality Testing
        Access to: ndir.getRequiredFeatureShape(), ndir.feedForward(featureSet, inference)
        featureSet size must be either:
        3d: [popSize, 1, featureInputLength]
        1d (broadcast): [featureInputLength]
        """
        # print(f"\n\nTS: {ts} -> {ts + 1}")
        
        # grab slices of each feature texture for this frame
        xFrame = self.x_2d[tsIndex, :] # all: [popSize]
        xDotFrame = self.xDot_2d[tsIndex, :]
        thetaFrame = self.theta_2d[tsIndex, :]
        thetaDotFrame = self.thetaDot_2d[tsIndex, :]
        
        
        # -------- FORCE DECISIONS --------
        # cat in order
        thisFrame = torch.cat([
            xFrame.view([1, -1]), 
            # xDotFrame.view([1, -1]),
            thetaFrame.view([1, -1])
            # thetaDotFrame.view([1, -1])
        ], dim=0) # [features, popSize]
        # print(f"{thisFrame}")
        
        # transpose
        thisFrame = thisFrame.t() # [popSize, features]
        # print(f"{thisFrame}")
        
        # reshape
        thisFrame = thisFrame.reshape([popSize, 1, 2]) # [popSize, 1, 4]
        # print(f"{thisFrame}")
        
        # feed forward, need size [pop, 1, numFeatures: 4]
        # ndir.getRequiredFeatureShape()
        trainResults = ndir.feedForward(thisFrame, inference=False).squeeze() # [popSize]
        # print(f"res: {trainResults}")
        # print(f"{ndir.grid.textureCrate['0_lstm_long_mem']}")
        
        # -------- CART PHYSICS --------
        # insert directly into next ts except for bounds
        [
            xPos,
            self.xDot_2d[tsIndex + 1, :],
            thetaRot,
            self.thetaDot_2d[tsIndex + 1, :],
            self.force_2d[tsIndex + 1, :]
            
        ] = self.cartPhysics(
            xFrame, 
            xDotFrame, 
            thetaFrame, 
            thetaDotFrame,
            trainResults
        )
        
        # self.worldTable()
        
        # physics bounds: x pos clamp and unrotate theta
        # hardTanh = torch.nn.Hardtanh(-20., 20.)
        # self.x_2d[tsIndex + 1, :] = hardTanh(xPos)
        self.theta_2d[tsIndex + 1, :] = self.unrotateTheta(thetaRot)
        
        # clamp x by nullifying x acceleration if outside of -20, 20
        X_CLAMP_POS = 60
        X_CLAMP_NEG = -60
        self.x_2d[tsIndex + 1, :] = torch.where(xPos > X_CLAMP_POS, X_CLAMP_POS, xPos)
        self.x_2d[tsIndex + 1, :] = torch.where(xPos < X_CLAMP_NEG, X_CLAMP_NEG, xPos)
        self.xDot_2d[tsIndex + 1, :] = torch.where(xPos > X_CLAMP_POS, 0, self.xDot_2d[tsIndex + 1, :]) #-10
        self.xDot_2d[tsIndex + 1, :] = torch.where(xPos < X_CLAMP_NEG, 0, self.xDot_2d[tsIndex + 1, :]) #10
        
        # self.worldTable()
        
        """
        !! there is a non-issue where the last 2 frames in the end
        will always copy each other because they are shallow refs, and
        only get detached in the next frame
        but just slice of 0th and -1st frame and everything should be dandy
        
        another issue is that x clamp does not nullify x acceleration
        """
    
    
    @timing
    def cartPhysics(self,
        x_1d: torch.Tensor,
        xDot_1d: torch.Tensor,
        theta_1d: torch.Tensor,
        thetaDot_1d: torch.Tensor,
        force_2d: torch.Tensor
    ) -> list[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        """
        Take in 5 tensors (current attributes: x, xdot, theta, thetadot, force)
        return arr of attributes in same order size 5 with performed cart physics
        all are size: [self.popSize], (numTimesteps must be sliced)
        """
        TOTAL_MASS = 1.1
        LENGTH_POLE = .5
        POLE_MASS_LENGTH = .05 # ???
        TAU = .02 # time between
        
        # ~~ trig ~~
        costheta = torch.cos(theta_1d)
        sintheta = torch.sin(theta_1d)
        
        # ~~ temp ~~ # temp is a badly named var, but it came from the .c code
        temp = (thetaDot_1d**2) * POLE_MASS_LENGTH * sintheta
        temp += force_2d
        temp /= TOTAL_MASS
        
        # ~~ thetaacc ~~
        fourthirds = torch.ones_like(temp) * (4/3)
        at = (sintheta * 9.8) - (costheta * temp) # gravity=9.8
        bt = ((costheta**2) * .1) / TOTAL_MASS
        bt = (fourthirds - bt) * LENGTH_POLE
        thetaacc = at / bt
        
        # ~~ xacc ~~
        xacc = (costheta * thetaacc * POLE_MASS_LENGTH) / TOTAL_MASS
        xacc = temp - xacc
        
        # ~~ accumulate ~~
        x_1d += (TAU * xDot_1d)
        xDot_1d += (TAU * xacc)
        theta_1d += (TAU * thetaDot_1d)
        thetaDot_1d += (TAU * thetaacc)
        
        # catch a weird bug where the pole just keeps spinning???
        # free energy babyyyyy
        # give it theta velocity drag between pi/8 and -pi/8
        theta_cond = torch.abs(theta_1d)
        SLOW_AMT = .95
        thetaDot_1d = torch.where(theta_cond < .4, thetaDot_1d * SLOW_AMT, thetaDot_1d)
        
        return [
            x_1d,
            xDot_1d,
            theta_1d,
            thetaDot_1d,
            force_2d
        ]
    
    
    @timing
    def cosScore(self):
        """
        Cosine score a theta tensor size: [numTimesteps,popSize]
        (theta is rotated left 90 deg)
        score = max cosine achieved (good for beginning stages) + accumulated cosine (later stages)
        equal weights
        returns score size: [self.popSize]
        """
        # print(f"t2d: {self.theta_2d}")
        
        theta_2d = torch.nan_to_num(self.theta_2d)
        theta_2d = torch.cos(theta_2d)
        maxed = torch.max(theta_2d, dim=0).values
        summed = torch.sum(theta_2d, dim=0)
        # score_1d = maxed + summed
        score_1d = summed
        """
        print(f"{theta_2d}")
        print(f"{score_1d}")
        print(f"max: {maxed}")
        print(f"sum: {summed}")
        """
        
        # divide by total or half frames to make it a ratio instead of num
        self.score_1d = score_1d / numTimesteps
        # print(f"{self.score_1d}")
    
    
    @timing
    def unrotateTheta(self, theta_1d: torch.Tensor) -> torch.Tensor:
        """
        Anything outside of 1 rotation of theta will ruin network input regularization
        always unrotate it so theta between -pi to pi
        theta_1d: self.theta[current population theta vals, :]
        returns theta_1d
        """
        FULLROT = 2 * math.pi
        
        # above pi
        theta_1d = torch.where(theta_1d > math.pi, theta_1d - FULLROT, theta_1d)
        
        # below -pi
        theta_1d = torch.where(theta_1d <= -math.pi, theta_1d + FULLROT, theta_1d)
        
        return theta_1d
ssn = Session()
#ssn.graph.ax.set_ylim([-.2, 1])


# helper func tests
# @lambda _: _() # iife
def helperTests():
    """
    # print(f"{ndir.grid.textureCrateContents()}")
    c = torch.tensor([[2,-4,-5,6],[2,4,5,6],[2,4,5,-6]], **ndir.gconf)
    c = torch.reshape(c, [3, 1, -1])
    res = ndir.feedForward(c)
    print(f"{res}")
    """
    # ndir.getRequiredFeatureShape()
    # ndir.exportGrid()
    # ndir.grid.textureCrateContents()
    # print(f"{ndir.grid.textureCrate["act_dense_weight"]}")
    # c = torch.tensor([2,1], **ndir.gconf)
    # res = ndir.feedForward(c, inference=False)
    # print(f"{res}")
    fakeScore = torch.tensor([1,7,2,3,1,0], **ndir.gconf)
    ndir.evoStep(fakeScore)
    # print(f"{ndir.grid.textureCrate["act_dense_weight"]}")
    # ndir.grid.textureCrateContents()
    

# -------- LOOP --------
for gen in range(numGenerations):
    
    # -------- TEST --------
    ssn.resetSim() # new samples every generation
    for ts in range(numTimesteps - 1): ssn.trainTest(ts) # train
    ssn.printStartLoop(gen)

    # -------- SCORE --------
    ssn.cosScore()

    # -------- TEMPERATURE CONTROL --------
    ssn.interruptGPU()

    # -------- EVOLVE --------
    ndir.evoStep(ssn.score_1d) # evolution
    tGraph = ndir.getPerfGraphSlice(ssn.score_1d) # grab highest and lowest elite to graph
    ssn.timeTrackUpdate(ndir.getEvoTimeTracking()) # time tracking, this gets session stats too
    
    # -------- LOGIC TICK --------
    if gen % 10 == 0:
        ndir.exportGrid() # savestate
    
    # -------- GRAPH UPDATE, EXPORT, TIMING, & FREEZE --------
    ssn.updateGraphTensor(tGraph) # update internal tensor only
    if gen % 10 == 0: ssn.redrawGraph() # expensive redraw every %x logic ticks
    # ssn.redrawGraph() # DEBUG

# -------- TIMING STATS --------
ssn.timeTrackOutput()

# end session save
ndir.sessionSave(numGenerations, "included tensors: x (2d), theta (2d), force (2d), scores (1d)", {
    "x": ssn.x_2d,
    "theta": ssn.theta_2d,
    "force": ssn.force_2d,
    "scores": ssn.score_1d
})

ssn.redrawGraph()
ssn.freezeGraph()
"""
"""