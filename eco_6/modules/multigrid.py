"""
A grid is a collection of neural networks bundled into 3rd dimension for population
-
Dims: [0]=z=depth, [1]=y=height, [2]=x=width/length
Weights: z=population, y=prior neuron height, x=current neuron height
Biases: z=population, y=1, x=current neuron bias
-
4th-dim is synthetic and references exist as separate items in a list, but l=layer depth
Squash functions are static and will always be torch.Hardtanh(-2, 2)
-
Functionality will be added to later add more layer-depths, keeping complexity simple in the beginning
"""
import torch
import copy
import math
import eco_6.modules.savestate as savestate
from eco_6.eco_print import EcoPrint

class MultiGrid():
    def __init__(self,
        masterConfig,
        gconf
    ):
        """
        Init means only to set an object\n
        Weights get a random normal distribution\n
        Biases get 0s\n
        A grid is the whole term used for multiple networks used for the different members within a population
        --
        see multigrid.py for more details
        """
        self.popSize = masterConfig["sim"]["popSize"]
        self.gridcon = masterConfig["grid"]
        self.gconf = gconf
        self.currVal = -1 # starts as inputs, feeds forward
        
        self.e = EcoPrint()
        
        # textures/tensors reside in a dict so we can dynamically get and set
        # only for learnable weights and biases, no driver/loop textures (those go in problem driver's Session class)
        self.textureCrate = {}
        
        # named squash functions
        self.squash = {
            "linear":   lambda x: x,
            "hardtanh22": lambda x: torch.nn.functional.hardtanh(x, -2., 2.),
            "relu6":    lambda x: torch.nn.functional.relu6(x),
            "argmax":   lambda x: torch.argmax(x, dim=2), # dim=2 stays inside pop member
            "softmax":  lambda x: torch.softmax(x, dim=2),
            
            # lstm types
            "scalar":   lambda x: torch.nn.functional.hardtanh(x, -1., 1.),
            "gate":     lambda x: torch.nn.functional.hardtanh(x, 0., 1.),
        }


    # -------- GRID CREATION, FEEDING --------
    def createGrid(self):
        """
        Create a new grid, textures go in self.textureCrate[<prefix_type_weight/bias>]
        all textures are 3d
        ex. 0_dense_weights, act_dense_bias, 1_lstm_xt_weights, ...
        """
        # errorize
        if len(self.gridcon["layers"]) == 0: print("config.json5 was given 0 layers to build network with")
        
        priorHeight = self.gridcon["featureInputLength"]
        for li, layer in enumerate(self.gridcon["layers"]):
            prefix = self.getLayerPrefix(li)
            
            # -------------------- DENSE --------------------
            if not layer["memory"]:
                
                self.textureCrate[f"{prefix}_dense_weight"] = torch.randn( # weights
                    [self.popSize, priorHeight, layer["height"]], **self.gconf
                )
                self.textureCrate[f"{prefix}_dense_bias"] = torch.zeros( # biases
                    [self.popSize, 1, layer["height"]], **self.gconf
                )
                
                
            # -------------------- LSTM --------------------
            elif layer["memory"] == "lstm":
                self.textureCrate[f"{prefix}_lstm_short_mem"] = torch.zeros( # short memory
                    [self.popSize, 1, layer["height"]], **self.gconf
                )
                self.textureCrate[f"{prefix}_lstm_long_mem"] = torch.zeros( # long memory
                    [self.popSize, 1, layer["height"]], **self.gconf
                )
                
                self.textureCrate[f"{prefix}_lstm_xt_weights"] = torch.randn( # xt weights / prev layer
                    [self.popSize, priorHeight, layer["height"] * 4], **self.gconf
                )
                self.textureCrate[f"{prefix}_lstm_sm_weights"] = torch.randn( # short mem weights
                    [self.popSize, 4, layer["height"]], **self.gconf
                )
                
                self.textureCrate[f"{prefix}_lstm_bias"] = torch.zeros( # biases
                    [self.popSize, 4, layer["height"]], **self.gconf
                )
                
            priorHeight = layer["height"]
            
        # -------------------- DROPOUT --------------------
        # dropout masks exist in texture crate now
        if 1.0 > self.gridcon["dropout"] > 0.0: self.refreshDropoutMask()
    
    def resetMemory(self):
        """
        reset long & short lstm memory
        should be done every load since memory starts building from 0
        """
        for li, layer in enumerate(self.gridcon["layers"]):
            prefix = self.getLayerPrefix(li)
            if layer["memory"] == "lstm":
                self.textureCrate[f"{prefix}_lstm_short_mem"][:] = 0
                self.textureCrate[f"{prefix}_lstm_long_mem"][:] = 0
    
    def refreshDropoutMask(self):
        """
        totally refreshes dropout factor mask (see paper)
        dropout mask size [currentHeight]
        contains a 0 in the neuron that should get nulled
        contains a 1/1-rate multiplier
        ex. .75 drop rate might result in [1.34, 0, 1.34, 1.34] for pop 3, curr height 4
        dropmask is universal for all pop members
        """
        for li, layer in enumerate(self.gridcon["layers"]):
            prefix = self.getLayerPrefix(li)
            
            actualDropped = math.floor(self.gridcon["dropout"] * layer["height"]) # int
            if prefix == "act": # non final layer dropped
                actualDropped = 0
                
            mult = 1.0 / (1.0 - (actualDropped / layer["height"])) # grab actual droprate based on actual
            # normal formula is 1 / (1 - droprate)
            
            res = torch.randperm(layer["height"], **self.gconf)
            res = torch.where(res < actualDropped, 0.0, mult)
            self.textureCrate[f"{prefix}_dropmask"] = res

    def checkFeatureShape(self, tensor):
        """
        Check and set the feature inputs\n
        Tensor MUST either be:\n
        1d size=[featureLengthInput]\n
        OR:\n
        3d size=[popSize, 1, featureLengthInput]\n
        Use 1d if the same information is being distributed to all members every ts step\n
        Use 3d if each member has a totally separate information in tracking, such as the polecart test
        """
        # check dimensions
        if len(tensor.size()) == 1:
            
            # 1d, only check length is == self.gridcon["featureInputLength"]
            if tensor.size()[0] == self.gridcon["featureInputLength"]:
                # passes
                self.currVal = tensor
            else:
                # err
                self.e.errorize("MultiGrid.setFeatures() 1d ran into wrong size")
                self.e.errorize(f"size given={tensor.size()[0]} -- featureInputLength={self.gridcon["featureInputLength"]}\n")
            
        elif len(tensor.size()) == 3:
            
            # 3d, need to check dim1 is flat and dim0==self.popSize and dim2==feature length
            sizeReq = torch.Size([self.popSize, 1, self.gridcon["featureInputLength"]])
            if tensor.size() == sizeReq:
                # passes
                self.currVal = tensor
            else:
                # err
                self.e.errorize("MultiGrid.setFeatures() 3d ran into wrong size")
                self.e.errorize(f"size given={tensor.size()} -- size needed={sizeReq}\n")
            
        else:
            
            # err
            self.e.errorize(f"MultiGrid.setFeatures() ran into false dimensionality: num dim={len(tensor.size())} -- must be 1 or 3")

    def getRequiredFeatureShape(self):
        """Return required 1d or 3d size for feature sizes"""
        print(f"getRequiredFeatureShape():")
        print(f"\t1d: [{self.gridcon["featureInputLength"]}] (broadcasted, similar features for all)")
        print(f"\t3d: [{self.popSize}, 1, {self.gridcon["featureInputLength"]}] (different input features for any)")

    def feedForward(self, inference: bool):
        """inference True: live/validation, False: training && dropout rate"""
        dropoutApplicable = (self.gridcon["dropout"] > 0.0) and (inference == False)
        
        for li, layer in enumerate(self.gridcon["layers"]):
            prefix = self.getLayerPrefix(li)
            
            # -------------------- DENSE --------------------
            if not layer["memory"]:
                # matmul and reshape (since it auto flattens)
                self.currVal @= self.textureCrate[f"{prefix}_dense_weight"]
                self.currVal = torch.reshape(self.currVal, [self.popSize, 1, -1])
                
                # bias
                self.currVal += self.textureCrate[f"{prefix}_dense_bias"]
                
                # dropout
                if dropoutApplicable:
                    self.currVal *= self.textureCrate[f"{prefix}_dropmask"]
                
                # squash
                thisActivation = layer["squash"]
                self.currVal = self.squash[thisActivation](self.currVal)
                
            # -------------------- LSTM --------------------
            elif layer["memory"] == "lstm":
                # xt matmul & reshape
                self.currVal @= self.textureCrate[f"{prefix}_lstm_xt_weights"]
                self.currVal = torch.reshape(self.currVal, [self.popSize, 4, -1])
                
                # short mem repeat, element mult
                tempVal = self.textureCrate[f"{prefix}_lstm_short_mem"]
                tempVal = tempVal.repeat(1, 4, 1)
                tempVal *= self.textureCrate[f"{prefix}_lstm_sm_weights"]
                
                # add bias & short mem
                self.currVal += self.textureCrate[f"{prefix}_lstm_bias"]
                self.currVal += tempVal
                
                # squash gate (relu1), scalar (hardtanh11)
                self.currVal[:, 0:3, :] = self.squash["gate"](self.currVal[:, 0:3, :]) # domain y-dim:[0, 3)
                self.currVal[:, 3, :] = self.squash["scalar"](self.currVal[:, 3, :]) # domain y-dim:[3]
                
                # long mem
                self.textureCrate[f"{prefix}_lstm_long_mem"] *= self.currVal[:, 0:1, :] # forget
                self.currVal[:, 1, :] *= self.currVal[:, 3, :] # input mult
                self.textureCrate[f"{prefix}_lstm_long_mem"] += self.currVal[:, 1:2, :] # input to long
                
                # reassign short mem
                self.textureCrate[f"{prefix}_lstm_short_mem"] = self.textureCrate[f"{prefix}_lstm_long_mem"]
                
                # apply dropout here
                if dropoutApplicable:
                    self.textureCrate[f"{prefix}_lstm_short_mem"] *= self.textureCrate[f"{prefix}_dropmask"]
                
                # output scalar squash & mult
                self.textureCrate[f"{prefix}_lstm_short_mem"] = \
                    self.squash["scalar"](self.textureCrate[f"{prefix}_lstm_short_mem"]) # output squash
                self.textureCrate[f"{prefix}_lstm_short_mem"] *= self.currVal[:, 2:3, :] # mult
                
                # store to next layer
                self.currVal = self.textureCrate[f"{prefix}_lstm_short_mem"]
            
            else: print(f"grid.feedForward() ran into invalid layer memory type: {layer["memory"]}")
    
    # -------- GET / SET --------
    def getDeepCopyGrid(self) -> dict[torch.Tensor]:
        """Returns a deep copied textureCrate"""
        return copy.deepcopy(self.textureCrate)
    
    def overwriteGrid(self, setTensor: torch.Tensor):
        """Overwrite current textureCrate/grid"""
        self.textureCrate = setTensor

    def getL2RegPenalty(self, lambdaMult: float = 1.0) -> torch.Tensor:
        """
        Get L2 Ridge Penalty
        lambdaMult: weight importance of output
        future: possibly add a pop normalizing mode
        --
        Likely to modify score with
        self.score = (original fitness score) - L2Penalty
        Note that fitness score is inverse loss (higher is better)
        """
        totalSum = 0
        
        # loop thru and find just weight textures
        for key, texture in self.textureCrate.items():
            wbType, _ = key.split("_")
            if wbType == "wt": # is a weight
                # print(f"{key}")
                # print(f"origin:\n{texture}")
                
                # square texture
                sq = texture**2 # texture=[popSize, prior nh, curr nh]
                #print(f"sqd:\n{sq}")
                
                # cannot just use sum because that would sum across popSize too
                
                # check dim 2 can support sum, length=1 will err
                if sq.size()[2] > 1:
                    sq = torch.sum(sq, dim=2) # [popSize, dim2 sum]
                    
                #print(f"dim2:\n{sq}")
                
                # check dim 1
                if sq.size()[1] > 1:
                    sq = torch.sum(sq, dim=1) # sum by population member [popSize]
                
                #print(f"dim1:\n{sq}")
                totalSum += sq
        
        
        return totalSum * lambdaMult


    # -------- DEBUG / OUTPUT --------
    def getLayerPrefix(self, li):
        # if last layer, act_ is name
        prefix = li
        if li == (len(self.gridcon["layers"]) - 1): prefix = "act"
        return prefix
    
    def gridSizeOutput(self):
        """Grid size as it exists in texture crate"""
        
        msg = f"\npopSize: {self.popSize}\n"
        msg += f"\t[feat]\t={self.gridcon["featureInputLength"]}\n"
        for li, layer in enumerate(self.gridcon["layers"]):
            prefix = self.getLayerPrefix(li)
            msg += f"\t[{prefix}]\t+{layer["height"]}, squash: {layer["squash"]}, mem: {layer["memory"]}\n"
        return msg

    def textureCrateContents(self):
        """Peek into the texture crate"""
        
        # keys
        tcKeys = [i for i in self.textureCrate.keys()]
        tcKeys = ", ".join(tcKeys)
        self.e.cyan(f"keys: {tcKeys}\n")
        
        # key value elaboration
        [print(f"--------------------------\n{i[0]}\n{i[1]}") for i in self.textureCrate.items()]


    # -------- IMPORT EXPORT --------
    def verifyStat(self, driverStat: int|str, fileStat: int|str, statName: str):
        """When loading a grid, verify driver stat matches imported one from file"""
        if driverStat == fileStat:
            # equal = fine
            return fileStat
        else:
            # unequal = problem
            errStr = f"MultiGrid.import....Grid() had unequal {statName}! Driver gave {driverStat} but size in savestate was {fileStat}.\n"
            infoStr = f"Change {statName} in driver to {fileStat}\n\n"
            self.e.errorize(errStr)
            self.e.warn(infoStr)
            return False
    
    def importGrid(self):
        """Import a MultiGrid (whole population) with stats"""
        imported = savestate.Import("population.tcdata")
        
        # retrieve stats other than popSize
        self.gridcon["featureInputLength"] = imported["stats"]["featureInputLength"]
        self.gridcon["layers"] = imported["stats"]["layers"]
        
        # verify popSize
        self.popSize = self.verifyStat(self.popSize, imported["stats"]["popSize"], "popSize")
        
        # set texturecrate
        self.textureCrate.clear()
        self.textureCrate = imported["crate"]
        
    def exportGrid(self):
        """Export a MultiGrid with grid stats"""
        gridToSave = copy.deepcopy(self.textureCrate)
        
        exportDict = {
            "stats": {
                "popSize": self.popSize,
                "featureInputLength": self.gridcon["featureInputLength"],
                "layers": self.gridcon["layers"]
            },
            "crate": gridToSave
        }
        
        # savestate export
        savestate.Export(
            exportDict, 
            filename="population", 
            fileExt=".tcdata",
            version=2.0
        )