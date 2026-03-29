#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the probability of aquiring an item via crafting versus earning 
enough to purchase the same item in Pathfinder 2e or Startfinder 2e

Created on Fri Mar 27 12:02:43 2026
@author: steve
"""

import numpy as np
import math
from functools import reduce
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
                    prog='craftProb',
                    description=__doc__,
                    epilog='')
parser.add_argument('-g', '--cost', type=float, default=250, help='full cost of item being crafting/purchased')
parser.add_argument('-l', '--level', type=int, default=2, help='level of the item')
parser.add_argument('-c', '--characterLevel', type=int, default=3, help='level of the character')
parser.add_argument('-m', '--modifier', type=int, default=11, help='craft modifier')
parser.add_argument('-p', '--proficiency', type=int, default=2, help='crafting proficiecy 1:Trained 2:Expert 3:Master 4:Legendary')
parser.add_argument('-r', '--rolls',type=int, default=3, help='number of consecutive downtime checks')
parser.add_argument('-d', '--days',type=int, default=8, help='nubmer of crafting days available per check')
parser.add_argument('-u', '--noCrafting', action='store_true', help='earn money to purchase instead of craft')
parser.add_argument('-f', '--noFormula', action='store_true', help='craft without having the formula for the item')
parser.add_argument('-v', '--rollsPerLevel',type=int, default=3, help='number of downtime checks per levelup')
parser.add_argument('-o', '--levelOffset',type=int, default=0, help='offset on the rolls per level check (closer to leveling up)')
args = parser.parse_args()

itemCost = args.cost # full cost of item being crafted/purchased
itemLevel = args.level # level of item
characterLevel = args.characterLevel # level of character doing the crafting/earning
craftModifier = args.modifier # craft modifier of character
proficiencyLevel = args.proficiency # 1 = trained, 2 = expert, 3 = master, 4 = legendary
maxDepth = args.rolls # consecutive rolls (each adds prep time, each x20 the size of the arrays)
maxDays = args.days # number of days spent in downtime (8 for OPF) without requiring a new roll/prep
useCrafting = not args.noCrafting # True to craft, False to earn
haveFormula = not args.noFormula # True if have the formula, False if not (still need access to example)
rollsPerLevel = args.rollsPerLevel # number of checks done before typical level up (3 for OPF)
levelOffset = args.levelOffset # offset for current character level (i.e. already made X rolls/checks)
proficiencies = ['Untrained','Trained','Expert','Master','Legendary']
EarnTable = np.array([[   1,   1,   1,   1,   1],
                      [   1,   2,   2,   2,   2],
                      [   1,   3,   3,   3,   3],
                      [   1,   5,   5,   5,   5],
                      [   1,   7,   8,   8,   8],
                      [   2,   9,  10,  10,  10],
                      [   3,   5,  10,  10,  10],
                      [   4,  20,  25,  25,  25],
                      [   5,  25,  30,  30,  30],
                      [   6,  30,  40,  40,  40],
                      [   7,  40,  50,  60,  60],
                      [   8,  50,  60,  80,  80],
                      [   9,  60,  80, 100, 100],
                      [  10,  70, 100, 150, 150],
                      [  15,  80, 150, 200, 200],
                      [  20, 100, 200, 280, 280],
                      [  25, 130, 250, 360, 400],
                      [  30, 150, 300, 450, 550],
                      [  40, 200, 450, 700, 900],
                      [  60, 300, 600,1000,1300],
                      [  80, 400, 750,1500,2000],
                      [ 100, 500, 900,1750,3000]])
DCTable = np.array([14,15,16,18,19,20,22,23,24,26,27,28,30,31,32,34,35,36,38,39,40,42,44,46,48,50])
FormulaCost = np.array([5,10,20,30,50,80,130,180,250,350,500,700,1000,1500,2250,3250,5000,7500,12000,20000,35000])

if proficiencyLevel < 1 or proficiencyLevel < 3 and itemLevel > 8 or proficiencyLevel < 4 and itemLevel > 16:
    exit(f"You cannot craft an {itemLevel} level item as a {proficiencies[proficiencyLevel]}")

#amountEarned = np.zeros(tuple([20 for x in range(maxDepth)]),dtype=float)
#amountCrafted = np.zeros(tuple([20 for x in range(maxDepth)]),dtype=float)

# arrays for storing results of choosing crafting
daysSpentCrafting = np.zeros(tuple([20 for x in range(maxDepth)]),dtype=int) # days already spent
costLeftCrafting = itemCost/2 * np.ones(tuple([20 for x in range(maxDepth)]),dtype=float) # cost remaining
craftRolls = levelOffset * np.ones(tuple([20 for x in range(maxDepth)]),dtype=int) # rolls made

# arrays for storing results of choosing earn income
daysSpentEarning = np.zeros(tuple([20 for x in range(maxDepth)]),dtype=int) # days already spent
costLeftEarning = itemCost/2 * np.ones(tuple([20 for x in range(maxDepth)]),dtype=float) # cost remaining
buyRolls = levelOffset * np.ones(tuple([20 for x in range(maxDepth)]),dtype=int) # rolls made

def earnOnRoll(natRoll,craft=False,haveFormula=True,modifier=0,taskLevel=0,proficiencyLevel=1,days=8):
    myEarn = 0 
    myDC = DCTable[taskLevel]
    myRoll = natRoll + modifier + (natRoll < 2 and -10 or natRoll > 19 and 10 or 0)
    #print(myRoll,myDC)
    days -= craft and haveFormula and 1 or craft and not haveFormula and 2 or 0
    if myRoll < myDC-10:
        pass
    elif myRoll < myDC:
        if craft:
            pass
        else:
            myEarn += days * EarnTable[taskLevel,0]
    elif myRoll < myDC+10:
        myEarn += days * EarnTable[taskLevel,proficiencyLevel]
    else:
        myEarn += days * EarnTable[taskLevel+1,proficiencyLevel]
    return(myEarn)

def daysForItem(natRoll,
                daysSpent=0,
                costLeft=0,
                rolls=0,
                craft=False,
                haveFormula=True,
                modifier=0,
                taskLevel=2,
                levelUpTask=False,
                characterLevel=0,
                levelUpCharacter=True,
                proficiencyLevel=1,
                maxDays=8,
                itemCost=None):
    """
    Use to calculate number of days to complete or purchase an item
    
    Parameters
    ----------
    natRoll : int
        natural roll of the dice. (1 to 20)
    daysSpent : int, optional
        number of days spent  crating/earning thus far. The default is 0.
    costLeft : float, optional
        cost left to account for via crafting/earning. The default is 0.
    rolls : int, optional
        number of craft checks (roll) made thus far. The default is 0.
    craft : boolean, optional
        calculate assuming crafting (not earning). The default is False.
    haveFormula : boolean, optional
        you have the formula for the item. The default is True.
    modifier : int, optional
        crafting modifier. The default is 0.
    taskLevel : int, optional
        task level for calculating DCs. The default is 2.
        This should the same as the Item Level for crafting
            and the Charcter Level - 2 for earning
    levelUpTask : boolean, optional
        level up the task level every 3 rolls. The default is False.
    characterLevel : int, optional
        character level for determining value crafted/earned. The default is 0.
        This should be Charcter Level for crafting but Character Level - 2 for earning
    levelUpCharacter : boolean, optional
        level up the task level every 3 rolls. The default is True.
    proficiencyLevel : int, optional
        1 = trained, 2 = expert, 3 = master, 4 = legendary. The default is 1.
    maxDays : int, optional
        number of downtime days per check. The default is 8.
    itemCost : float, optional
        full cost of item. The default is None.

    Returns
    -------
    (daysSpent,costLeft,rolls+1) : (int,float,int)

    """
    if costLeft <=0: return(daysSpent,costLeft,rolls+1) 
    tl = taskLevel + (levelUpTask and int(rolls/rollsPerLevel) or 0) # level up task
    m = modifier + (levelUpCharacter and int(rolls/rollsPerLevel) or 0) # level up modifier
    cl = characterLevel + (levelUpCharacter and int(rolls/rollsPerLevel) or 0) # level up character
    if tl < 0: tl = 0 # cannot go less than 0
    if cl < 0: cl = 0 # cannot go less than 0
    myDC = DCTable[tl] # get DC of the task
    myRoll = natRoll + m + (natRoll < 2 and -10 or natRoll > 19 and 10 or 0) # modifiy roll for crits
    #print(myRoll,myDC)
    # Calculate the prep days if crafting with or without a formula
    prepDays = 0 
    prepDays += craft and haveFormula and 1 or craft and not haveFormula and 2 or 0 
    if myRoll < myDC-10:
        # crit fail earns nothing
        days = maxDays - prepDays
        earn = 0
        # model the loss of raw materials by increasing the cost
        # this is not technically accurate, but it will due for now
        if (craft and itemCost and itemCost > 0):
            costLeft += itemCost*0.05 # 10% or raw materials (half the cost) is lost on crit fail
    elif myRoll < myDC:
        # fail earn nothing if crafting
        if craft:
            days = maxDays - prepDays
            earn = 0
        else:
            # still get something if earning income
            days = math.ceil(costLeft/EarnTable[cl,0])
            earn = EarnTable[cl,0]
    elif myRoll < myDC+10:
        # success
        days = math.ceil(costLeft/EarnTable[cl,proficiencyLevel])
        earn = EarnTable[cl,proficiencyLevel]
    else:
        # crit success
        days = math.ceil(costLeft/EarnTable[cl+1,proficiencyLevel])
        earn = EarnTable[cl+1,proficiencyLevel]
    #print(days,earn)

    # calculate the daysSpent and costLeft
    if (prepDays + days > maxDays):
        daysSpent += maxDays
        costLeft -= (maxDays-prepDays)*earn
    else:
        daysSpent += prepDays + days
        costLeft -= days*earn

    return(daysSpent,costLeft,rolls+1)

def consecutiveRolls(natRolls,
                     daysSpent=0,
                     costLeft=0,
                     rolls=0,
                     craft=False,
                     haveFormula=True,
                     modifier=0,
                     taskLevel=2,
                     levelUpTask=False,
                     characterLevel=0,
                     levelUpCharacter=True,
                     proficiencyLevel=1,
                     maxDays=8,
                     itemCost=None):
    """
     Recursive function to calculate number of days to complete or purchase 
     an item based on a tuple of dice rolls
     
     Parameters
     ----------
     natRolls : tuple of int
         natural rolls of the dice. (1 to 20)
     The rest of the prameters are the same as those from daysForItem

    Returns
    -------
    (daysSpent,costLeft,rolls+1) : (int,float,int)

    """
    for natRoll in natRolls:
        (daysSpent,costLeft,rolls) = daysForItem(natRoll,
                                                      daysSpent,
                                                      costLeft,
                                                      rolls,
                                                      craft,
                                                      haveFormula,
                                                      modifier,
                                                      taskLevel,
                                                      levelUpTask,
                                                      characterLevel,
                                                      levelUpCharacter,
                                                      proficiencyLevel,
                                                      maxDays,
                                                      itemCost)
    return(daysSpent,costLeft,rolls)

#with np.nditer(amountEarned, flags=['multi_index'], op_flags=['readwrite']) as ae:
#    for a in ae:
#        #print("%d <%s>" % (a, ae.multi_index), end=' ')
#        a[...] = reduce(lambda x,y: x + earnOnRoll(y+1,False,False,craftModifier,characterLevel-2,proficiencyLevel,8), ae.multi_index, 0)  
#with np.nditer(amountCrafted, flags=['multi_index'], op_flags=['readwrite']) as ae:
#    for a in ae:
#        #print("%d <%s>" % (a, ae.multi_index), end=' ')
#        a[...] = reduce(lambda x,y: x + earnOnRoll(y+1,True,True,craftModifier,itemLevel,proficiencyLevel,8), ae.multi_index, 0)  

# fill the arrays with the reslut of the roll combinations for crafting
#  the array indices are used to create every natural roll result
with np.nditer(daysSpentCrafting, flags=['multi_index'], op_flags=['readwrite']) as ae:
    for a in ae:
        #print("%d <%s>" % (a, ae.multi_index), end=' ')
        theseRolls = tuple([ x+1 for x in ae.multi_index ])
        a[...],costLeftCrafting[ae.multi_index],craftRolls[ae.multi_index] = \
            consecutiveRolls(theseRolls, 
                             a, 
                             costLeftCrafting[ae.multi_index],
                             craftRolls[ae.multi_index],
                             True,
                             haveFormula,
                             craftModifier,
                             itemLevel,
                             False,
                             characterLevel,
                             True,
                             proficiencyLevel,
                             maxDays,
                             itemCost)
    
# fill the arrays with the results of the roll combinations for earning
#  the array indices are used to create every natural roll result
with np.nditer(daysSpentEarning, flags=['multi_index'], op_flags=['readwrite']) as ae:
    for a in ae:
        #print("%d <%s>" % (a, ae.multi_index), end=' ')
        theseRolls = tuple([ x+1 for x in ae.multi_index ])
        a[...],costLeftEarning[ae.multi_index],buyRolls[ae.multi_index] = \
            consecutiveRolls(theseRolls, 
                             a, 
                             costLeftEarning[ae.multi_index],
                             buyRolls[ae.multi_index],
                             False,
                             haveFormula,
                             craftModifier,
                             characterLevel-2,
                             True,
                             characterLevel-2,
                             True,
                             proficiencyLevel,
                             maxDays,
                             None)
            
myCraft = daysSpentCrafting.reshape(reduce(lambda x,y:x*y,daysSpentCrafting.shape,1))
myBuy = daysSpentEarning.reshape(reduce(lambda x,y:x*y,daysSpentEarning.shape,1))
myHist = np.stack((myCraft,myBuy),1)
fig,ax = plt.subplots()
#ax.hist(myHist,maxDepth*8+1,histtype='bar',label=('craft','buy'))
ax.hist(myHist,histtype='bar',density=True,align='right',label=('craft','buy'))
ax.legend(prop={'size': 10})
myTitle = f"Craft/Buy {itemCost/2} cred of a level {itemLevel} item \n"
myTitle += f"over {maxDepth} downtime checks of {maxDays} days each \n"
myTitle += f"character level {characterLevel+levelOffset} to {characterLevel+levelOffset+int((maxDepth-1)/3)}, "
myTitle += f"craft({proficiencies[proficiencyLevel][0]}) "
myTitle += f"mod {craftModifier+levelOffset} to {craftModifier+levelOffset+int((maxDepth-1)/3)} "
ax.set_title(myTitle)
ax.set_xlabel('days')
xticklabels = [x.get_text() for x in ax.get_xticklabels()]
xticklabels[-1] += '+'
xticklabels[-2] += '+'
#print(ax.get_xticks(),xticklabels)
ax.set_xticks(ax.get_xticks()[0:-1],xticklabels[0:-1])
#low,high = ax.get_xlim()
#ax.set_xlim(low,high-1)
plt.show()
