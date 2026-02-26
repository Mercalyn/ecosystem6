"""
Global static 12 col wide, unlimited rows makes making quick interfaces simple
One object per interface window
-
dev:
missing tons of primitives
need to make a lef col navigation thing that can append to row.. or maybe a vertical rows option.
or could do a vertical tabs
"""
import customtkinter

class Window:
    # --------------------------- init ---------------------------
    def __init__(
            self, 
            title:str="Default Window", 
            size:tuple=(1200, 800)
        ):
        customtkinter.set_appearance_mode("dark") # "dark", "light", or "system"
        
        # app frame
        self.frame = customtkinter.CTk()
        self.frame.title(title)
        self.frame.geometry(f"{size[0]}x{size[1]}")
        
        # give all 12 cols a weight of 1
        allCols = tuple([i for i in range(12)])
        self.frame.grid_columnconfigure(allCols, weight=1)
        
        # row data contains an asymmetric 2d array containing rows, then col within that row of actual ctk objects
        # this is so functionality can exist in my class, not thru ctk classes
        self.rowData = []
        
        
    # --------------------------- primitives ---------------------------
    prim = {
        # ~~~~ organization ~~~~
        "container": None,
        "containerScroll": None,
        "tab": None,
        
        # ~~~~ output ~~~~
        "label": lambda frame, colItem: customtkinter.CTkLabel(frame, text=colItem["text"]),
        "progressBar": lambda frame, colItem: customtkinter.CTkProgressBar(frame, orientation="horizontal"),
        "graphBar": None,
        "graphLine": None,
        "graphScatter": None,
        
        # ~~~~ input ~~~~
        "button": lambda frame, colItem: customtkinter.CTkButton(frame, text=colItem["text"]),
        "check": None,
        "textSingle": None,
        "textMulti": None,
        "menu": None,
        "radio": None,
        "slider": None,
        "switch": None,
    }
    
    
    # --------------------------- util ---------------------------
    def addRow(self, iterElement:list) -> None:
        """
        Pass in an array of tuples containing (element, start col, num cols to span)
        ex. iterElement = addRow([{ #[0][0]
            "element": "progressBar",
            "colStart": 1,
            "colSpan": 10
        },{ #[0][1]
            "element": "label",
            "colStart": 11,
            "colSpan": 1,
            "text": ".04%"
        }])
        """
        
        # get current row from rowData
        currRow = len(self.rowData)
        innerColArr = []
        
        for colItem in iterElement:
            """
            colItem = { # anything with * see primitives for what each element takes
                element:string,
                colStart,
                colSpan,
                text, #*
            }
            """
            
            # find primitive
            thisElement = self.prim[colItem["element"]]
            
            # execute primitive, giving it the frame and the dict
            thisElement = thisElement(self.frame, colItem)
            
            # set default layout, can be changed by accessing .rowData[y][x] ...
            thisElement.grid(
                row=currRow, 
                column=colItem["colStart"], 
                padx=4, #.padx
                pady=4, #.pady
                sticky="ew",  #.anchor
                columnspan=colItem["colSpan"]
            )
            
            innerColArr.append(thisElement)
            
        # give current row an array of variables
        self.rowData.append(innerColArr)
    
    
    def debugMode(self, mode:bool) -> None:
        """Run this before run() to have it list out [row][col] coordinates"""
        if mode:
            for yIndex, yVal in enumerate(self.rowData):
                for xIndex, xVal in enumerate(yVal):
                    #print(yIndex, xIndex)
                    try:
                        # can change text attr
                        self.rowData[yIndex][xIndex].configure(text=f"[ {yIndex} ][ {xIndex} ]")
                    except ValueError:
                        # has no text attr, such as progressBar
                        pass
    
    
    def run(self):
        #print(f"{self.rowData}")
        
        # mainloop
        self.frame.mainloop()
        