"""
see '0 File types, structure'.txt for spec for version 2.0
"""

import datetime
import torch

class Export:
    def __init__(self, savestateData, filename: str = "", fileExt: str = ".tcdata", version: float = 2.0):
        """
        Export a MultiGrid in a savestate\n
        uses pytorches ability to save models and dicts\n
        ex. filename = "savestate"\n
        savestateData should have the format listed at the top of savestate.py\n
        ex. fileExt=".tcdata", include the .
        also seen in file extensions.txt
        """
        
        # time format ex: 12Aug2024--1542.tcdata
        # local machine time
        d = datetime.datetime.now()
        dateStr = f"{d.day}{d.strftime("%b")}{d.year}--{d.strftime("%H")}{d.strftime("%M")}"
    
        # extension
        filepath = f"{filename}{fileExt}"
        
        # implement date and version
        savestateData["version"] = version
        savestateData["date"] = dateStr
        # print(f"{savestateData}")
        
        # save whole thing
        torch.save(savestateData, filepath)


class Import:
    def __new__(cls, filepath): # new cls is like init but can return
        '''
        Import a MultiGrid in a savestate\n
        ex. filepath="12Aug2024--1542.tcdata"\n
        filepath="savestate.tcdata"\n
        filepath="gen_0.s4"
        '''
        
        # open
        loaded_data = torch.load(filepath)
        print(f"Imported grid from {loaded_data['date']}")
        
        return loaded_data