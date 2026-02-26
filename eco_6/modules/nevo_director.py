"""
Neuro Evolution Director
"""
import torch
import json5
import os # only for debug clear terminal
from pathlib import Path
from eco_6.modules.multigrid import MultiGrid
from eco_6.modules.evolution import Evolution
import eco_6.modules.savestate as savestate
from eco_6.eco_print import EcoPrint
torch.autograd.set_grad_enabled(False)
from deepmerge import always_merger


class NevoDirector:
    # --------------------------- INIT ---------------------------
    def __init__(self,
        device,
        dtype,
        confo: list = [] 
    ):
        """
        Initialize the Neuro Evolution Director\n
        pass in named base config override found in "config.json5" located
        in same directory as driver problem .py
        """
        os.system("cls") # clear terminal
        
        
        # ---------------- CONFIG FILES ----------------
        self.masterConfig = {}
        
        # base config
        thisPath = Path(__file__).parent.parent # relative path to /eco_6
        with open(thisPath / "config/sim.json5", "r") as openFile:
            self.masterConfig["sim"] = json5.load(openFile)["sim"]
        with open(thisPath / "config/grid.json5", "r") as openFile:
            self.masterConfig["grid"] = json5.load(openFile)["grid"]
        with open(thisPath / "config/evo.json5", "r") as openFile:
            self.masterConfig["evo"] = json5.load(openFile)["evo"]
        
        # load overrides
        oConfig = {}
        with open("config.json5", "r") as openFile:
            oConfig = json5.load(openFile)
        for cf in confo:
            if not cf in oConfig: print(f"NevoDirector.__init__() conf override {cf} does not exist")
            always_merger.merge(self.masterConfig, oConfig[cf]) # deep nest-merges
        
        # print(json5.dumps(self.masterConfig, indent=4)) # print final config
        
        
        # ---------------- STARTUP ----------------
        # color print
        self.e = EcoPrint()
        self.e.info("GPU startup")
        self.e.dgrey(" ... ")
        torch.set_printoptions(sci_mode=False) # turn off the damn science mode
        
        # cuda check
        print(f"Found device[0]: {device}", end="")
        self.e.dgrey(" ... ")
        self.e.okay() if device == torch.device("cuda") else self.e.errorize(msg=": should be cuda")
        
        # easy type device settings access from anywhere, used as **self.gconf inside tensor args
        self.gconf = {
            "dtype": dtype, # change this one setting to reflect across entire system
            "device": device
        }
        
        # init graphing util
        
        # give evo config
        self.evo = Evolution(
            self.masterConfig,
            self.gconf
        )
        
        # half precision speedups (not implemented, lives in sim["allowHalfPrecision"])
        
        
        # ---------------- POPULATING ----------------
        # all methods give config, only "load" won't use since it overwrites with file stats
        self.grid = MultiGrid(
            self.masterConfig,
            self.gconf
        )
        
        # ~~ NEW ~~
        if(self.masterConfig["sim"]["populate"] == "new"): self.grid.createGrid() # new random gen from grid
        
        # ~~ LOAD ~~
        elif(self.masterConfig["sim"]["populate"] == "load"): 
            # overwrite self.masterConfig["sim"]["populate"]d grid with a whole file
            self.grid.importGrid()
            self.grid.resetMemory() # reset lstm memory
        
        # ~~ ERRORIZE ~~
        else: self.e.errorize(f"eco.evo.Evolution() tried to init with repopulate:str set to {self.masterConfig["sim"]["populate"]}")
        self.e.white(f"grid neuron size by layer: {self.grid.gridSizeOutput()}")
        self.e.dgrey("... ")
        self.e.okay() if self.grid.textureCrate != {} else self.e.errorize(msg=": grid did not init properly")
        print()
        """
        """
    
    
    # --------------------------- TIMESTEP ---------------------------
    def feedForward(self, featureInputs: torch.Tensor, inference: bool = False) -> torch.Tensor:
        """
        High-Level setFeatures & Feed Forward\n
        inference=False (default) if training, =True if inference/test/validation
        Returns actionspace [popSize, actions]
        """
        self.grid.checkFeatureShape(featureInputs)
        self.grid.feedForward(inference)
        #print(f"{self.grid.currVal}\n")
        return self.grid.currVal
    
    
    # --------------------------- MEMBER EVOLUTION ---------------------------
    def evoStep(self, scoreTexture_1d: torch.Tensor):
        """
        Automatically evolves all weights and biases found in self.grid.textureCrate\n
        based on evolution config\n
        Pass in ssn.score (or a 1d score texture) from main problem driver code
        --
        will also reset lstm memory
        """
        # ~~ DESTINATION MASK ~~
        self.evo.createDestinationMask(scoreTexture_1d)
        # print(f"{self.evo.destinationMask}")
        # debug known dest mask
        # self.evo.destinationMask = torch.tensor([10, 20, 80, 70, 0, 55], **self.gconf).view([-1, 1, 1])
        
        # ~~ EVOLVE ~~ based on destination mask
        for label, originTexture in self.grid.textureCrate.items():
            
            # check eligibility: any weight or bias
            if ("weight" in label) or ("bias" in label):
            # if label == "act_dense_weight": # DEBUG
                mergeTexture = torch.zeros_like(originTexture, **self.gconf) # feed evo outputs into here
                # print(f"b4 {originTexture}")
                
                mergeTexture = self.evo.opTourney(originTexture, scoreTexture_1d)
                mergeTexture = self.evo.opCross(originTexture, mergeTexture)
                mergeTexture = self.evo.opFork(originTexture, mergeTexture)
                mergeTexture = self.evo.opReroll(originTexture, mergeTexture)
                mergeTexture = self.evo.opEliteStayover(originTexture, mergeTexture)
    
                # assign real grid
                self.grid.textureCrate[label] = mergeTexture
        
        # ~~ REFRESH DROPOUT IF APPLICABLE ~~
        if 1.0 > self.masterConfig["grid"]["dropout"] > 0.0: self.grid.refreshDropoutMask()
        
        # reset lstm memory
        self.grid.resetMemory()
        
        # print(f"af {self.grid.textureCrate['act_dense_weight']}")
        self.e.okay()
        
    
    # --------------------------- UTILS ---------------------------
    def exportGrid(self): self.grid.exportGrid()
    def getRequiredFeatureShape(self): self.grid.getRequiredFeatureShape()
    def getPerfGraphSlice(self, scoreTexture): return self.evo.getPerfGraphSlice(scoreTexture)
    
    def getL2Penalty(self, lambdaMult: float = 1.0) -> torch.Tensor:
        """High-level ridge penalty return"""
        return self.grid.getL2RegPenalty(lambdaMult)
        
    def getEvoTimeTracking(self) -> dict:
        """Get time tracking stats, 1 call works for both evo and session stats"""
        return self.evo.getTimeTrackedObjs()
    
    def sessionSave(self, 
        genNum: int = 0, 
        addInfo: str = "no info given", 
        trackableTextures: dict = {}
    ):
        """
        In order to animate a timestep / test session,
        Export a dict of all trackable textures to file to replay it in a different script / animator
        """
        exportedSession = {
            "stats": {
                # empty for future use
            },
            "info": addInfo,
            "trackable": trackableTextures
        }
        
        savestate.Export(
            exportedSession, 
            filename=f"gen_{genNum}", 
            fileExt=".s4",
            version=2.0
        )