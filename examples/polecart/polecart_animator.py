"""
Polecart Animator / Polecart Grapher
for the polecart sessions
lets graph it 1st
then use pygame to visualize it
"""
# -------- IMPORTS --------
import sys
sys.path.append("../..")
import eco_6.ecosys as eco
import torch
import pygame
import sys
import math
import os # only for debug clear terminal
from eco_6.graph import MultiLineGraph
gpu = torch.device(type="cuda")
os.system("cls")

# consts
MEM_INDEX = 33 # 0-5999 please note that members are out-of-order in regards to score
GEN_NO = 160 # file name only

# load session
s4 = eco.evo.savestate.Import(f"gen_{GEN_NO}.s4")
print(f"INFO: {s4["info"]}")
# print(f"{s4["trackable"]["x"].size()}") # ex. x = [120, 6] = [TS, pop]

# DEBUG::find highest score
scores = s4["trackable"]["scores"]
maxIndex = torch.argmax(scores)
minIndex = torch.argmin(scores)
print(f"max {scores[maxIndex]} @ {maxIndex}\nmin {scores[minIndex]} @ {minIndex}")

# select all timesteps from member -> 1d each
x = s4["trackable"]["x"][:, MEM_INDEX]
theta = s4["trackable"]["theta"][:, MEM_INDEX]
force = s4["trackable"]["force"][:, MEM_INDEX]
# print(f"{x}")

"""
# get one pop member flat -> 2d
memberSession = posRotVel[:, MEM_INDEX, :]
memberForce = xforce[:, MEM_INDEX]
oneScore = scores[MEM_INDEX]
print(f"score: {oneScore}")
#print(f"memberForce: {memberForce}")

# xpose x:y so it is [4, TS_RANGE]
memberSession = memberSession.T
#print(f"mses c:\n{memberSession}")


xaxis = torch.Tensor.cpu(xaxis)
memberSession = torch.Tensor.cpu(memberSession)
memberForce = torch.Tensor.cpu(memberForce)
"""

# create xaxis
xaxis = torch.arange(0, x.size()[0], dtype=torch.float64)
# print(f"{xaxis}")

# convert to arr
x = x.tolist()
theta = theta.tolist()
force = force.tolist()
# print(f"{x}")

graph = MultiLineGraph(
    x_axis_data=xaxis,
    y_axis_data_arr=[
        x,
        theta,
        force
    ],
    legend=[
        "x",
        "theta",
        "force"
    ],
    y_label="..",
    x_label="step",
    graph_title="trackable over time"
)
graph.freeze_window()
"""
"""


# pygame animate
pygame.init()
screen = pygame.display.set_mode((1600, 800))
pygame.display.set_caption("Polecart Animator")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

frameCount = len(x)
currFrame = 0
#print(f"{frameCount}")

while True:
    if currFrame > frameCount - 1:
        currFrame = 0
    
    # check events first for player input
    for event in pygame.event.get():
        # event::quit
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    
    pygame.draw.rect(screen, (0,0,0), (0, 0, 1600, 800))
    
    # frame updates
    # -------- CART --------
    # x=-60 to 60
    xf = x[currFrame]
    #print(f"{x}")
    # convert to pixels 100-1100
    xf += 60 # 0-120
    xf *= 12.5 # 0-1500
    xf += 50 # 50-1550
    cartRect = pygame.Rect(0, 0, 60, 30)
    cartRect.center = (xf, 700)
    pygame.draw.rect(screen, (255, 0, 0), cartRect)
    # draw track
    trackRect = pygame.Rect(0, 0, 1600, 10)
    trackRect.center = (800, 715)
    pygame.draw.rect(screen, (255, 255, 255), trackRect) # track
    
    
    # -------- POLE --------
    poleRect = pygame.Rect(0, 0, 5, 100)
    poleRect.center = (xf, 635)
    poleSurf = pygame.Surface((5, 100))
    poleSurf.fill((255, 255, 0))
    poleSurf.set_colorkey((0,0,0))
    # rotation math and shit
    thetaf = -theta[currFrame]
    xOffset = math.sin(thetaf) * 50
    yOffset = math.cos(thetaf) * 50
    deg = math.degrees(thetaf)
    rotated = pygame.transform.rotate(poleSurf, deg) # positive = CCW
    newCoord = rotated.get_rect(center = poleSurf.get_rect(topleft = (xf - xOffset, 630 - yOffset)).center)
    screen.blit(rotated, newCoord)
    
    
    # -------- FORCE --------
    innerForce = force[currFrame]
    # separate forces to L/R
    lForce = -innerForce if innerForce < 0 else 0
    rForce = innerForce if innerForce > 0 else 0
    # scale # starts at 0-6
    lForce *= 33.34 # 0-200
    rForce *= 33.34
    lfRect = pygame.Rect(0, 0, lForce, 5)
    rfRect = pygame.Rect(0, 0, rForce, 5)
    lfRect.center = (xf - (lForce / 2) - 28, 700)
    rfRect.center = (xf + (rForce / 2) + 28, 700)
    pygame.draw.rect(screen, (255, 0, 180), lfRect) # force left
    pygame.draw.rect(screen, (255, 0, 180), rfRect) # force right
    
    
    # -------- TEXT --------
    # frame
    frameText = font.render(f"frame: {currFrame} / {frameCount} ({.02 * frameCount} secs)", True, (255, 255, 255))
    screen.blit(frameText, (20, 20))
    # gen
    genText = font.render(f"generation: {GEN_NO}", True, (255, 255, 255))
    screen.blit(genText, (20, 44))
    # member id
    memText = font.render(f"member id: {MEM_INDEX}", True, (255, 255, 255))
    screen.blit(memText, (20, 68))
    # ending score
    thisScore = scores[MEM_INDEX] * 100
    scoreText = font.render(f"ending score: {thisScore:.1f}%", True, (255, 255, 255))
    screen.blit(scoreText, (20, 92))
    
    
    # -------- UPDATE --------
    # update with .flip() or .update()
    pygame.display.update()
    
    # max fps
    clock.tick(50) # sim was 50
    currFrame += 1
"""
"""