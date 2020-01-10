#load csv
import numpy as np
from random import shuffle
import itertools
import datetime
import random

costs = np.genfromtxt('distances.csv', delimiter=',')
print(costs)



def rouletteWheelSelection(population):
    max = sum([agent.cost for agent in population])
    probs = [agent.cost/max for agent in population]
    return population[np.random.choice(len(population), p=probs)]

class Gene:
    path = []
    cost = 0.0

    def __init__(self, path):
        self.path=path
        self.cost = pathCost(costs, path)

def kmToNm(lengthInKm):
    return lengthInKm*0.5399565


def NmTokm(lenghtInNm):
    return lenghtInNm*1.852001

def genPath():
    #paths generated without start and end
    #start and end added in path cost calculation
    #path = [0]
    path_ = []
    choices = [1, 2, 3, 4, 5, 6, 7, 8]
    shuffle(choices)
    path_.extend(choices)
    #path_.append(0)
    return path_

def genetic(populationSize=50, generations=20, crossover=95, mutation=1,printResults=True):
    population = []
    newPath = []
    #fill population
    for i in range(populationSize):
        population.append(Gene(genPath()))
    for k in range(generations):
        #crossover etc
        #sort population
        population.sort(key=lambda x: x.cost, reverse=False)
        for path in population:
            if random.randint(1,101)<=crossover:
                #random recomb
                #gen1, gen2 =  random.sample(population, k=2)
                gen1 = rouletteWheelSelection(population)
                gen2 = rouletteWheelSelection(population)
                #print(gen1, gen2)
                for i in range(8):
                    coinToss = random.randint(0, 2)
                    if coinToss == 0:
                        g = gen1.path[i]
                    else:
                        g = gen2.path[i]
                    #insertion in path list

                    if g in newPath:
                        #print(g, "with cointoss ", coinToss, " exists!")
                        if coinToss == 0:
                            if gen2.path[i] in newPath:
                                while True:
                                    g=random.randint(1,8)
                                    if g not in newPath:
                                        #newPath.append(g)
                                        break
                            else:
                                g = gen2.path[i]

                        elif coinToss == 1:
                            if gen1.path[i] in newPath:
                                while True:
                                    g = random.randint(1, 8)
                                    if g not in newPath:
                                        #newPath.append(g)
                                        break
                            else:
                                g = gen1.path[i]

                    newPath.append(g)
                #print(pathCost(costs, newPath), path.cost)
                if pathCost(costs, newPath)<path.cost:
                    #print("REPLACED")
                    path.path = list(newPath)
                    #print("New Path:", path)
            elif random.randint(1,101)<=mutation:
                path.path = genPath()
            
            newPath.clear()

    if(printResults):
        print("Genetic Algorithm")
        print("Best path = ", printPath(population[0].path))
        print(
            "Best path length in kilometres = % .2f [km]" % population[0].cost)
        print("Best path length in Nautical Miles = % .2f [NM]" % kmToNm(
            population[0].cost))
        print("-----")

    #return

    return (population[0].path, population[0].cost)

def simulatedAnnealing(populationSize=50, printResults=True):
    population = []
    #fill population
    for i in range(populationSize):
        population.append(Gene(genPath()))

    #g = Gene(genPath())
    T = 100
    alpha = 0.000001
    k=0
    for g in population:
        while T>0:
            #select neighbour solution
            #switch values of 2 indices

            a = random.randint(1,7)
            b = random.randint(1,7)
            tmpPath = g.path
            tmpPath[a], tmpPath[b] = tmpPath[b], tmpPath[a]
            g2 = Gene(tmpPath)
            
            if g2.cost<g.cost:
                g = g2
            elif random.randint(1,101)<T:
                g = g2
            T -= k*alpha
            k+=1
            #print(T)
    population.sort(key=lambda x: x.cost, reverse=False)
    if(printResults):
        print("Simulated Annealing")
        print("Best path = ", printPath(population[0].path))
        print("Best path length in kilometres = % .2f [km]" % population[0].cost)
        print("Best path length in Nautical Miles = % .2f [NM]" % kmToNm(population[0].cost))
        print("-----")

    return (population[0].path, population[0].cost)


def printPath(path):
    pathString = r"Porat"
    for p in path:
        pathString += r" $\rightarrow$ "
        if p == 1:
            pathString += r"Ceja"
        if p == 2:
            pathString += r"Bodula\v{s}"
        if p == 3:
            pathString += r"Levan"
        if p == 4:
            pathString += r"Fenera"
        if p == 5:
            pathString += r"Porti\'{c}"
        if p == 6:
            pathString += r"Golumbera"
        if p == 7:
            pathString += r"Fenoliga"
        if p == 8:
            pathString += r"Porer"
    pathString += r" $\rightarrow$ Porat"
    pathString += "\n"
    return pathString

def pathCost(costs, path,startEnd=0):
    cost = 0
    cost += costs[startEnd][path[0]]
    for i in range(len(path)-1):
        cost += costs[path[i]][path[i+1]]
    cost += costs[path[i+1]][startEnd]
    return cost

def extensiveSearch(printResults=True):
    bestCost = 100.0
    bestPath = []
    choices = [1, 2, 3, 4, 5, 6, 7, 8]
    allPaths = list(itertools.permutations(choices, 8))
    print("Length of all paths=",len(allPaths))
    #print(allPaths)
    #add beggining and end to the path
    for path in allPaths:
        path = list(path)
        path.insert(0,0)
        path.append(0)
    #calculate costs and add best
    for path in allPaths:
        #print(path)
        currCost = pathCost(costs, path)
        if currCost <= bestCost:
            bestCost = currCost
            bestPath = path[:]
    if(printResults):
        print("Extensive search")
        print("Best path = ", printPath(bestPath))
        print("Best path length in kilometres = % .5f [km]" % bestCost)
        print("Best path length in Nautical Miles = % .2f [NM]" % kmToNm(bestCost))
        print("-----")



#print(pathCost(costs, path2))
#print(genPath())
#print(pathCost(costs, genPath()))


# extensiveSearchStartTime = datetime.datetime.now()
# extensiveSearch()
# extensiveSearchEndTime = datetime.datetime.now()

# extensiveSearchTimeLength = extensiveSearchEndTime-extensiveSearchStartTime

# print("Time needed for extensive search = ", extensiveSearchTimeLength, "[s]")
"""
geneticStartTime = datetime.datetime.now()
genetic()
geneticEndTime = datetime.datetime.now()

geneticTimeLength = geneticEndTime - geneticStartTime

print("Time needed for genetic algorithm = ", geneticTimeLength, "[s]")

saStartTime = datetime.datetime.now()
simulatedAnnealing()
saEndTime = datetime.datetime.now()

saTimeLength = saEndTime - saStartTime

print("Time needed for simulated annealing = ", saTimeLength, "[s]")
"""
avgSA = []
avgGA = []

t_avgSA = []
t_avgGA = []

runs = 100
"""
for i in range(runs):
    print(i, "% done.", end="\r")
    startTime=datetime.datetime.now()
    ga_path, ga_cost = genetic(printResults=False)
    t_avgGA.append(datetime.datetime.now()-startTime)
    startTime = datetime.datetime.now()
    sa_path, sa_cost = simulatedAnnealing(printResults=False) 
    t_avgSA.append(datetime.datetime.now()-startTime)
    avgGA.append(ga_cost)
    avgSA.append(sa_cost)

# #print(avgGA)

tt_avgGA = []
tt_avgSA = []
for x in range(len(t_avgGA)):
    tt_avgGA.append(t_avgGA[x].microseconds/1000000.0)
    tt_avgSA.append(t_avgSA[x].microseconds/1000000.0)

"""

# print("Mean best result for Genetic Algorithm over ", runs," runs: ", np.mean(avgGA), "[km], with standard deviation of: ", np.std(avgGA))
# print("Mean best result for Simulated Annealing over ", runs, " runs: ", np.mean(avgSA), "[km], with standard deviation of: ", np.std(avgSA))
# print("Mean time for Genetic Algorithm over ", runs, " runs: ",
#       np.mean(t_avgGA), "[s], with standard deviation of: ", np.std(tt_avgGA))
# print("Mean time for Simulated Annealing over ", runs, " runs: ",
#       np.mean(t_avgSA), "[s], with standard deviation of: ", np.std(tt_avgSA))
# print("Minimum GA:", min(avgGA))
# print("Minimum SA:", min(avgSA))
# print("Maximum GA:", max(avgGA))
# print("Maximum SA:", max(avgSA))

from matplotlib import pyplot as plt
choices = [1, 2, 3, 4, 5, 6, 7, 8]
allPaths = np.array(list(itertools.permutations(choices, 8)))
allPathCosts = []
for x in allPaths:
    allPathCosts.append(kmToNm(pathCost(costs, x)))
allPathCosts.sort()
allPathCostsNP=np.array(allPathCosts)
#print(allPathCostsNP.shape)
#allPaths.sort()
print(allPathCostsNP.shape)

yy_=np.mean(allPathCostsNP)

print("Mean: ", yy_)

xx_=np.arange(0,40320, 1)
horiz_line_data = np.array([yy_ for i in range(40320)])
#horiz_line_best = np.array([24.51 for i in range(40320)])
horiz_line_meanGA = np.array([15.5 for i in range(40320)])
horiz_line_meanSA = np.array([15.38 for i in range(40320)])

print(horiz_line_data.shape)
print(allPathCostsNP.shape)
print(xx_.shape)        
plt.figure()
plt.style.use('grayscale')
plt.ylim(0,32)
plt.grid()

plt.plot(xx_, allPathCostsNP, label="sorted path lengths", linewidth=3)

plt.title("Individual Path lengths")
plt.xlabel("Path")
plt.ylabel("Path length [NM]")
plt.plot(horiz_line_data, '--', label="mean path length", linewidth=3)
plt.plot(horiz_line_meanGA, ':', label="mean result - GA", linewidth=3)
plt.plot(horiz_line_meanSA, '-.', label="mean result - SA", linewidth=3)
plt.legend(loc='upper left')
plt.savefig("tst3.png")
