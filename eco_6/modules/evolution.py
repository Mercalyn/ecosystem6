"""
Evolution will create a new population given an old population and a score
--
evolution methods:
~~ tourneyCross ~~ not implemented
semi-bracket battle for best score, chooses best tourneyBestRate% of the time
and then finalists cross. lower best rate eases selection pressure, but tourneyCross
is the 2nd form of selection pressure

~~ tourney ~~
as of ecosys_6 uses a max-compression efficient algo for tournaments
used as the primary selection pressure
finalist battle for best score, no crossover

~~ cross ~~
simple crossover from parent A and B using a mean=0 var=1 normal, below 0=from A

~~ reroll ~~
totally new member, easiest way to keep fresh genes entering the pool

~~ stayover ~~
stays to next simulation due to random chance, this is good for easing
selection pressure and increasing uniqueness in the face of pure competition

~~ fork ~~
remainder of non chosen. rates given as a 1=100%, generally .02 or .005
hardfork will choose a completely different value for a gene
softfork will nudge in a direction default of mean=0, variance=softforkMult


--
for elitism:
update: elitism has been phased out in favor of multiple tournament passes.
elitism a lot of times you get too much copying,
destroying the creativity and nn diversity that makes it work so well.

1-member elitism is enabled by default so it at least keeps the very top solution.
this also allows the graph to not "lose" its best solution appearing like loss of progression

--
Evolution Codes (because tensor cannot contain enums)
INCOMPLETE  = -1

ELITE       =  0
TOURNEY     = 10
CROSSOVER   = 20
REROLL      = 55
STAYOVER    = 70
FORK        = 80
"""
import torch
from eco_6.eco_print import EcoPrint
import random
from eco_6.timing import timing, getTimeTrackedObjs


class Evolution:
    # --------------------------- init ---------------------------
    def __init__(self, 
        masterConfig, # master config loaded from config & overrides
        gconf         # gpu device and dtype
    ):
        """
        Evolution init = config only\n
        """
        self.evocon = masterConfig["evo"]
        self.popSize = masterConfig["sim"]["popSize"]
        self.gconf = gconf
        self.getTimeTrackedObjs = getTimeTrackedObjs # allows outside scripts to get evo time trackings
        
        self.e = EcoPrint()
        self.destinationMask = None
        
        # need to add an int gconf for reindexing and masking
        self.gconf_int = {
            "dtype": torch.int,
            "device": self.gconf["device"]
        }
        self.gconf_int8 = {
            "dtype": torch.int8,
            "device": self.gconf["device"]
        }

    @timing
    def createDestinationMask(self, scoreTexture_1d: torch.Tensor):
        """
        destinationMask determines the evolve method per population member
        destinationMask size: [popSize, 1, 1]
        scoreTexture_1d size: [popSize]
        """
        # check wrong size
        if scoreTexture_1d.size()[0] != self.popSize: print(f"ndir.evo.createDestinationMask() ran into false sizes: score: {scoreTexture_1d.size()[0]} -- popSize: {self.popSize}")
        
        rollRates = self.evocon["rolls"]
        mask = torch.ones([self.popSize], **self.gconf_int8)
        
        # give 20 rolls
        # randDistro starts with roll domain, turns to 99 for completed
        randDistribute = torch.randint(
            low=0, high=20, # domain=[0, 20)
            size=[self.popSize], **self.gconf_int8
        )
        
        # 1-member elitism by changing topmost to 0 (do not evolve)
        # avoid more than a single elite: too many causes problems
        argmaxIndex = torch.argmax(scoreTexture_1d)
        mask[argmaxIndex] = 0
        randDistribute[mask == 0] = 99 # elites done
        
        # fork 80
        delin = rollRates["fork"] # delineation goes upward
        mask[randDistribute < delin] = 80 # mask to fork
        randDistribute[mask == 80] = 99 # fork done
        
        # stayover 70
        delin += rollRates["stayover"]
        mask[randDistribute < delin] = 70 # mask to stayover
        randDistribute[mask == 70] = 99 # stayover done
        
        # reroll 55
        delin += rollRates["reroll"]
        mask[randDistribute < delin] = 55 # mask to reroll
        randDistribute[mask == 55] = 99 # reroll done
        
        # cross 20
        delin += rollRates["cross"]
        mask[randDistribute < delin] = 20 # mask to cross
        randDistribute[mask == 20] = 99 # cross done
        
        # tourney 10
        delin += rollRates["tourney"]
        mask[randDistribute < delin] = 10 # mask to tourney
        randDistribute[mask == 10] = 99 # tourney done
        
        # any remaining incomplete converted to remainder: fork 80
        mask[randDistribute != 99] = 10
        
        # reshape from 1d to 3d size [self.popSize, 1, 1]
        mask = torch.reshape(mask, [self.popSize, 1, 1])
        
        # assign
        self.destinationMask = mask
        
        # DEBUG::set destmask to something known
        #self.destinationMask = torch.tensor([80, 80, 80, 80, 99, 99], **self.gconf).view([-1, 1, 1])

    @timing
    def opFork(self, 
        originTex: torch.Tensor,
        mergeTex: torch.Tensor
    ) -> torch.Tensor:
        """
        Fork -- CODE 80
        softfork: nudge gene
        hardfork: reroll gene
        minfork: minimize to smaller weight
        --
        """
        # -------- SOFTFORK --------
        # create softfork / nudge tensor
        nudge = torch.randn(originTex.size(), **self.gconf) # mean=0 var=1
        nudge *= self.evocon["rates"]["softforkMult"] # mean=0 var=softforkMult
        
        # create softfork chance tensor, full 3d size
        nudgeChance = torch.rand(originTex.size(), **self.gconf) # 0-1
        
        # mask
        nudge = torch.where(nudgeChance < self.evocon["rates"]["softforkRate"], nudge, 0) # chance to nudge
        # nudge = torch.where(self.destinationMask == 80, nudge, 0) # dest mask
        
        """
        
        # -------- HARDFORK --------
        # create reroll tensor
        reroll = torch.randn(tcItem.size(), **self.gconf)
        #if tcKey == "w_i": print(f"a: {reroll}\n")
        
        # create hardfork chance tensor
        rerollChance = torch.rand(tcItem.size(), **self.gconf) # 0-1
        
        # mask in chance to fork
        reroll = torch.where(rerollChance < self.hardforkRate, reroll, 0)
        #if tcKey == "w_i": print(f"b: {reroll}\n")
        
        # mask in destinationMask == fork code
        reroll = torch.where(self.destinationMask == 80, reroll, 0)
        #if tcKey == "w_i": print(f"c: {reroll}\n")
        
        # merge by where logic
        gridRef[tcKey] = torch.where(reroll != 0, reroll, gridRef[tcKey])
        
        return gridRef
        """
        
        # ret actual nudge where fork is ordered
        return torch.where(self.destinationMask == 80, originTex + nudge, mergeTex)

    @timing
    def opCross(self, 
        parentAData: torch.Tensor,
        mergeTex: torch.Tensor
    ) -> torch.Tensor:
        """
        Crossover -- CODE 20\n
        parentA: destination member, must have code 20
        parentB: other member to be crossed with, does not need code 20
        """
        
        # reindex
        parentBIndex = torch.randperm(n=self.popSize, **self.gconf_int)
        # create parentB, reindexing from parentAData (this texture), shuffled in a z:0:popSize axis
        parentBData = parentAData[parentBIndex]
        
        # here parentBData should be shuffled texture
        
        # 50:50 split & merge parents into B
        splitAB = torch.rand(parentAData.size(), **self.gconf) # 0-1
        parentBData = torch.where(splitAB > .5, parentAData, parentBData)
        
        # return crossed pop where ordered, rest 0s
        return torch.where(self.destinationMask == 20, parentBData, mergeTex)

    @timing
    def opTourney(self, 
        originTex: torch.Tensor,
        scoreTexture_1d: torch.Tensor
    ) -> torch.Tensor:
        """
        Tourney -- CODE 10
        https://algorithmafternoon.com/books/genetic_algorithm/chapter04/
        new:
        max-compression for each maxCompressions will create a reindexing and 
        store index of max value element and compress it for another loop or until end
        """
        
        # setup
        # priorIdx acts as the max-compressed indices
        priorIdx = torch.arange(0, self.popSize, **self.gconf_int) # [popSize]
        # init to unshuffled randperm
        # scoreTexture_1d acts as an anchor that should not change past here, only the Idx will change
        # grabbing/reindexing from this
        # print(f"p.sco {scoreTexture_1d}")
        # ensure score is 1d
        if len(scoreTexture_1d.size()) > 1: 
            self.e.err(f"ndir.evo.opTourney() ran into {scoreTexture_1d.size()} num of dims. should be 1.\n")
        
        # loop
        for _ in range(self.evocon["tournaments"]["maxCompressions"]):
            reindex = torch.randperm(self.popSize, **self.gconf_int)
            currIdx = priorIdx[reindex]
            currScore = scoreTexture_1d[currIdx]
            
            # argmax
            priorIdx = torch.where(currScore > scoreTexture_1d, currIdx, priorIdx)
        
        # here scoreTexture_1d[priorIdx] will contain max-compressed scores,
        # and priorIdx will contain max-compressed indices
        # print(f"{priorIdx}")
        
        # return a reindexed tourney slots, rest 0s
        return torch.where(self.destinationMask == 10, originTex[priorIdx], 0)

    @timing
    def opReroll(self, originTex: torch.Tensor, mergeTex: torch.Tensor) -> torch.Tensor:
        """Reroll -- CODE 55"""
        
        # create reroll tensor
        reroll = torch.randn(originTex.size(), **self.gconf)
        
        # merge
        return torch.where(self.destinationMask == 55, reroll, mergeTex)
    
    @timing
    def opEliteStayover(self, originTex: torch.Tensor, mergeTex: torch.Tensor) -> torch.Tensor:
        """Elite & Stayover -- CODE 0 & 70"""
        
        # merge
        merged = torch.where(self.destinationMask == 0, originTex, mergeTex)
        merged = torch.where(self.destinationMask == 70, originTex, merged)
        return merged
    
    @timing
    def getPerfGraphSlice(self, scoreTexture: torch.Tensor) -> torch.Tensor:
        """
        Return percentile of scores from population in this order:\n
        tensor1d=[100th percentile, 75th, 50th, 25th, 0th]
        --
        this function was using 70-95% of evolution time taken:
        it was because top = torch.max(castedScore).cpu() and .tolist() are very inefficient
        """
        # print(f"\n{scoreTexture}")
        
        # cast to float32 because quantile doesn't work on bfloat16 or float16
        castedScore = scoreTexture.float()
        
        """
        top = torch.quantile(castedScore, 1.0, interpolation="nearest", keepdim=True)
        bot = torch.quantile(castedScore, 0.0, interpolation="nearest", keepdim=True)
        """
        
        top = torch.max(castedScore, dim=0, keepdim=True).values
        bot = torch.min(castedScore, dim=0, keepdim=True).values
        
        # find 25, 50, 75th percentiles
        pc25 = torch.quantile(castedScore, .25, interpolation="nearest", keepdim=True)
        pc50 = torch.quantile(castedScore, .50, interpolation="nearest", keepdim=True)
        pc75 = torch.quantile(castedScore, .75, interpolation="nearest", keepdim=True)
        """
        print(f"{top}")
        print(f"{bot}")
        print(f"{pc50}")
        print(f"{pc75}")
        """
        
        # cat them into their own tensor slice [top, 75, 50, 25, bot]
        # to be stored in session_utils for graph chunking
        catSlice = torch.cat([top, pc75, pc50, pc25, bot], dim=0)
        # print(f"{catSlice}")
        
        return catSlice
