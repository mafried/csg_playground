import glob, os, shutil
import numpy as np
import copy


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



def getRunDataMinMax(directory):
    
    scoresList = []
    
    for runDir in next(os.walk(directory))[1]:
        statsFile = directory + "/" + runDir + "/stats.dat"        
        scores = getScores(statsFile)
        scoresList.append(scores)
    
    npa = np.array(scoresList)    
    minv = np.min(npa, axis=0)
    maxv = np.max(npa, axis=0)
    meanv = np.mean(npa, axis=0)
    
    return (minv, maxv, meanv)



def plotErrorBarForScores(resultPath):
    
    (mean, std) = getRunDataMeanAndStd(resultPath)
    iterations = np.arange(0,len(mean))

    html = '''
    <head>
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>

    <body>
    <div id="myDiv"><!-- Plotly chart will be drawn inside this DIV --></div>
    <script>
    

    var data = [ 
    {
      x: [
    '''
    for i in iterations:
        html = html + str(i) + ', '

    html = html + '''
    ],
    y: [
    '''

    for m in mean:
        html = html + str(m) + ', '
    
    html = html + '''
    ], 
    error_y: {
    type: 'data',
    visible: true, 
    array: [
    '''

    for s in std:
        html = html + str(s) + ', '

    html = html + '''
    ]
    }, 
    type: 'scatter'
    }
    ];
    Plotly.newPlot('myDiv', data);

      </script>
    </body>
    '''

    print(html)


def plotContinuousErrorBarForScores(resultPath):
    
    (minv, maxv, meanv) = getRunDataMinMax(resultPath)
    iterations = np.arange(0,len(meanv))

    html = '''
    <head>
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>

    <body>
    <div id="myDiv"><!-- Plotly chart will be drawn inside this DIV --></div>
    <script>


    var trace1 = {
    x: [
    '''
    for i in iterations:
        html = html + str(i) + ', '

    rev_iterations = iterations[::-1]
    for i in rev_iterations:
        html = html + str(i) + ','

    html = html + '''
    ],
    y: [
    '''

    for m in minv:
        html = html + str(m) + ', '

    rev_maxv = maxv[::-1]
    for m in rev_maxv:
        html = html + str(m) + ', '

    html = html + '''
    ], 
    fill: "tozerox",
    fillcolor: "rgba(0,100,80,0.2)",
    line: {color: "transparent"}, 
    type: "scatter",
    showlegend: false
    };

    var trace2 = {
    x: [
    '''
    for i in iterations:
        html = html + str(i) + ', '
    html = html + '''
    ],
    y: [
    '''

    for m in meanv:
        html = html + str(m) + ', '

    html = html + '''
    ], 
    line: {color: "rgb(0,100,80)"},
    mode: "lines",
    type: "scatter", 
    showlegend: false
    };

    var data = [trace1, trace2];

    var layout = {
    paper_bgcolor: "rgb(255,255,255)", 
    plot_bgcolor: "rgb(255,255,255)", 
    xaxis: {
    gridcolor: "rgb(125,125,125)", 
    showgrid: true, 
    linecolor: "rgb(125,125,125)", 
    showline: true, 
    showticklabels: true, 
    tickcolor: "rgb(125,125,125)", 
    ticks: "outside", 
    zeroline: false
    }, 
    yaxis: {
    gridcolor: "rgb(125,125,125)", 
    showgrid: true,
    linecolor: "rgb(125,125,125)", 
    showline: true, 
    showticklabels: true, 
    tickcolor: "rgb(125,125,125)", 
    ticks: "outside", 
    zeroline: false
    }
    };

    Plotly.newPlot('myDiv', data, layout);

      </script>
    </body>
    '''

    print(html)





resultPath = "../Release"
 
#plotErrorBarForScores(resultPath)
plotContinuousErrorBarForScores(resultPath)
