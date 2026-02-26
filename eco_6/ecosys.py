"""
Merc's library
ECOSYSTEM
.. because it takes a whole ecosystem
--
import eco_6.ecosys as eco
examples in ../examples/
OR
from eco_6.ecosys import db, gui, api, evo
--
STANDALONE:
graph.py -- class: multi line graph with dynamic updating
eco_print.py -- class: terminal with colors
timing.py -- func: timing decorator
"""


# --------------------------- database ---------------------------
"""
one eco.db.Table per table/db access
adb = eco.db.Table("test.db", "main")
"""
import eco_6.modules.database as db


# --------------------------- gui ---------------------------
"""
one eco.gui.Window per interface window(but we will have tab views soon, multiple things)
agui = eco.gui.Window("Welcome to Mass Scraper", size=(1200, 640))
"""
import eco_6.modules.interface as gui


# --------------------------- api ---------------------------
"""
one eco.api.Endpoint per different api endpoint/method
kanyeQuote = eco.api.Endpoint("https://api.kanye.rest")
"""
import eco_6.modules.api as api


# --------------------------- neuro evolution ---------------------------
"""
one eco.evo.NevoDirector per optimization problem
--
nevo_director.py master script with submodules:
-multigrid.py
-evolution.py
-nevo_graph.py
--
these scripts are super specific/organized, and one should copy an existing
script to start from a working base
"""
import eco_6.modules.nevo_director as evo
import eco_6.modules.session_utils as esu