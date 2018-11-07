import configparser
import os, glob, shutil, ntpath
import subprocess
import datetime
from shutil import copyfile
from collections import namedtuple

# Content path (not mandatory)
contentPath = "C:/Projekte/csg_playground_build/Release"

# Source of exe file
exePath = "C:/Projekte/csg_playground_build/Release"

# Specifies where the results should be stored. 
resultPath = "C:/Projekte/csg_playground_build/Release"

CSGConfig = namedtuple("CSGConfig", "points primitives partitioning algorithm config")
Run = namedtuple("Run", "csgConfig times")

def executeProg(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def config():
    return configparser.ConfigParser();

def run(run):
    
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    resultDir = resultPath + "/" + os.path.splitext(ntpath.basename(run.csgConfig.points))[0] + "_" +\
    os.path.splitext(ntpath.basename(run.csgConfig.primitives))[0] + "_" +\
    run.csgConfig.partitioning + "_" + run.csgConfig.algorithm + "_" + now 
    
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)    
        for i in range(0,run.times):
            runDir = resultDir + "/" + str(i)
            os.makedirs(runDir)
            
            # Copy exe
            copyfile(exePath + "/main_csg.exe", runDir + "/main_csg.exe")
            
            # Copy dlls
            files = glob.iglob(os.path.join(exePath, "*.dll"))
            for file in files:
                if os.path.isfile(file):
                    shutil.copy2(file, runDir)
            
            # Create ini file
            with open(runDir + "/params.ini", "w") as configfile:
                run.csgConfig.config.write(configfile)
                
            # Copy points file 
            copyfile(run.csgConfig.points, runDir + "/points.xyz")
                    
            # Copy primitives file
            copyfile(run.csgConfig.primitives, runDir + "/primitives.prim")
        

    for i in range(0,run.times):        
        try:
            runDir = resultDir + "/" + str(i)
            print("start " + str(i))
			
            args = [ runDir + "/main_csg.exe", runDir + "/points.xyz", runDir + "/primitives.prim",runDir + "/params.ini", run.csgConfig.partitioning, run.csgConfig.algorithm, "mesh"];
            print(args)
            with open(runDir + "/log.dat", "w") as f:
                for l in executeProg(args):
                    print(l, end="")
                    f.write(l)
            copyfile("./stats.dat", runDir + "/stats.dat")  
            copyfile("./mesh_tree.dot", runDir + "/mesh_tree.dot")
            copyfile("./connectionGraph.dot", runDir + "/connectionGraph.dot")
            copyfile("./mesh_mesh.obj", runDir + "/mesh_mesh.obj")
			
            os.remove("./stats.dat")
            os.remove("./mesh_tree.dot")
            os.remove("./connectionGraph.dot")
            os.remove("./mesh_mesh.obj")
		
				
            print("end")
        except subprocess.CalledProcessError as ex:
            print("execution failed. Reason: " + str(ex))

############################################################################################################
			
c = config()
c["Sampling"] = {
"MaxDistance": "0.02",
"MaxAngleDistance": "0.17",
"ErrorSigma": "0.01",
"ConnectionGraphSamplingStepSize": "0.05"
}
c["GA"] = {
"SimpleCrossoverRate":"1.0",
"CrossoverRate":"0.4",
"MutationRate":"0.3",
"InParallel":"true",
"UseCaching":"true",
"Cancellable":"false"
} 
c["Optimization"] = {
"OptimizationProb":"1.0",
"PreOptimizationProb":"1.0",
"OptimizationType":"traverse"
}
c["StopCriterion"] = {
"MaxIterations":"1000",
"MaxIterationsWithoutChange":"1000"
}
            
r = Run(csgConfig=CSGConfig(points=contentPath + "/model13.xyz", primitives=contentPath + "/model13.prim", partitioning="none", algorithm="ga", config=c), times=3)

run(r)

c["GA"] = {
"SimpleCrossoverRate":"0.0",
"CrossoverRate":"0.4",
"MutationRate":"0.3",
"InParallel":"true",
"UseCaching":"true",
"Cancellable":"false"
} 

run(r)

c["GA"] = {
"SimpleCrossoverRate":"1.0",
"CrossoverRate":"0.4",
"MutationRate":"0.3",
"InParallel":"true",
"UseCaching":"true",
"Cancellable":"false"
} 

run(r)

c["GA"] = {
"SimpleCrossoverRate":"1.0",
"CrossoverRate":"0.4",
"MutationRate":"0.3",
"InParallel":"true",
"UseCaching":"true",
"Cancellable":"false"
} 
c["Optimization"] = {
"OptimizationProb":"0.0",
"PreOptimizationProb":"0.0",
"OptimizationType":"traverse"
}

run(r)

c["GA"] = {
"SimpleCrossoverRate":"0.0",
"CrossoverRate":"0.4",
"MutationRate":"0.3",
"InParallel":"true",
"UseCaching":"true",
"Cancellable":"false"
} 
c["Optimization"] = {
"OptimizationProb":"0.0",
"PreOptimizationProb":"0.0",
"OptimizationType":"traverse"
}

run(r)
