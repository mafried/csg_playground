import glob, os, shutil
import statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def getScores(file):

    file = open(file)

    text = file.readlines()
    
    iterations = []
    bests = []
    
    for line in text:
        
        if(line.startswith("#")):
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
        
        iterations.append(float(parts[0]))
        bests.append(float(parts[1]))
       
        
    file.close()

    return bests# (iterations, bests)

def getRunDataMeanAndStd(directory):
    
    scoresList = []
    
    for runDir in next(os.walk(directory))[1]:
        statsFile = directory + "/" + runDir + "/stats.dat"        
        scores = getScores(statsFile)
        scoresList.append(scores)
    
    npa = np.array(scoresList)    
    mean = np.mean(npa, axis=0)
    std = np.std(npa, axis=0)   
    
    return (mean, std)

def plotErrorBarForScores(resultPath):
    
    (mean, std) = getRunDataMeanAndStd(resultPath)

    iterations = np.arange(0,len(mean))
    plt.xticks(iterations)
    
    #plt.errorbar(iterations, mean, yerr=std,fmt='o', color='black',
    #         ecolor='lightgray', elinewidth=3, capsize=0)

    plt.plot(iterations, mean)
        
    #plt.plot(iterations, mean - std)
    #plt.plot(iterations, mean + std)
    
    plt.fill_between(iterations, mean - std, mean + std,
                 color='gray', alpha=0.1)
    
    plt.show()
    

resultPath = "C:/Projekte/csg_playground_build/Release/model13_model13_none_ga_2018-11-06_172115"
 
plotErrorBarForScores(resultPath)