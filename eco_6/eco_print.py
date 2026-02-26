"""
separate entity from evolib, but useful
use with:
--
from eco_print import EcoPrint
eprint = EcoPrint()
--
possible colors
https://pypi.org/project/termcolor/
black 	    on_black
red 	    on_red
green 	    on_green
yellow 	    on_yellow
blue 	    on_blue
magenta 	on_magenta
cyan 	    on_cyan 	
white 	    on_white 	
light_grey 	on_light_grey 	
dark_grey 	on_dark_grey 	
light_red 	on_light_red 	
light_green 	on_light_green 	
light_yellow 	on_light_yellow 	
light_blue 	    on_light_blue 	
light_magenta 	on_light_magenta 	
light_cyan 	    on_light_cyan
"""
from termcolor import colored, cprint

class EcoPrint:
    """
    Merc's main color-terminal printing with tags\n
    Same line by default
    """
    # ~~ COLORS ~~
    def green(self, msg: str = ""):
        """Green Coloring"""
        print(colored(msg, "green"), end="")
    
    def yellow(self, msg: str = ""):
        """Yellow Coloring"""
        print(colored(msg, "yellow"), end="")
    
    def red(self, msg: str = ""):
        """Red Coloring"""
        print(colored(msg, "red"), end="")
    
    def lgrey(self, msg: str = ""):
        """Cyan Coloring"""
        print(colored(msg, "light_grey"), end="")
            
    def dgrey(self, msg: str = ""):
        """Cyan Coloring"""
        print(colored(msg, "dark_grey"), end="")
    
    def cyan(self, msg: str = ""):
        """Cyan Coloring"""
        print(colored(msg, "cyan"), end="")
    
    def magenta(self, msg: str = ""):
        """Magenta Coloring"""
        print(colored(msg, "magenta"), end="")
    
    def blue(self, msg: str = ""):
        """Blue Coloring"""
        print(colored(msg, "blue"), end="")
    
    def white(self, msg: str = ""):
        """White Coloring"""
        print(colored(msg, "white"), end="")
    
    
    # ~~ PREPEND TAGS ~~
    def info(self, msg: str = ""):
        """Yellow INFO prepend"""
        print(colored("INFO: ", "yellow") + msg, end="")
    
    def warn(self, msg: str = ""):
        """Yellow WARN prepend"""
        print(colored("WARN: ", "yellow") + msg, end="")
    
    def err(self, msg: str = ""):
        """Red ERR prepend"""
        print(colored("ERR: ", "red") + msg, end="")
    errorize = err
    
    def loop(self, msg: str = ""):
        """Cyan LOOP prepend"""
        print(colored("LOOP: ", "cyan") + msg, end="")
    
    def okay(self, msg: str = ""):
        """Green OK postpend"""
        print(msg + colored(" OK", "green") + "\n", end="")
        

if __name__ == "__main__":
    e = EcoPrint()
    e.blue("blue msg\n")
    e.errorize("err inline .. ")