#!/usr/bin/python2
from __future__ import division

"""
Loads a progress bar lasting for the passed in number of seconds
"""

import rospy
import time
import sys
import signal
import progressbar


# Handle sigterms if initiated by user
class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True


################## START OF SCRIPT ########################    
grace = GracefulKiller()

if len(sys.argv) != 2:
    print("Error incorrect number of arguments: " + str(sys.argv))
else:
    totalSeconds = int(sys.argv[1])

toolbar_width = 70
barPerSecs = toolbar_width / totalSeconds
barsAccumulated = 0
secondsPassed = 0    

bar = progressbar.ProgressBar(maxval=toolbar_width, \
                              widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

#totalSeconds = 30
# # setup toolbar
# sys.stdout.write("[%s]" % (" " * toolbar_width))
# sys.stdout.flush()
# sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

while not grace.kill_now and secondsPassed < totalSeconds:
  rospy.sleep(1) # do real work here
  secondsPassed += 1
  bars = int(barPerSecs * secondsPassed)
  while bars > barsAccumulated:
    bar.update(barsAccumulated)
    barsAccumulated += 1    
        # sys.stdout.write("#")
        # sys.stdout.flush()
sys.stdout.write("\n")
