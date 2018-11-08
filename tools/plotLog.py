# This is for plot the curve of logs of YOLO

import argparse 
import sys
import matplotlib.pyplot as plt

def main(argv):
    StartPoint = 20000#150000
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", help = "path to log file" )
    args = parser.parse_args()

    f = open(args.log_file) # open the logfile
    lines  = [line.rstrip("\n") for line in f.readlines()[StartPoint:]]
    
    numbers = {'1','2','3','4','5','6','7','8','9'}
    iters = []
    loss = []
    fig,ax = plt.subplots()
    prev_line = ""

    for line in lines:
        args = line.split(' ')
        if args[0][-1:]==':' and args[0][0] in numbers :
            iters.append(int(args[0][:-1]))         
            loss.append(float(args[2]))

    ax.plot(iters,loss)
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.grid()
    #ticks = range(0,250,10) 
    #ax.set_yticks(ticks)
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv)
