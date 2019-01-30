#!/usr/bin/python2
from __future__ import division

import rospy
import time
import sys
import signal


# Handle sigterms if initiated by user
class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

grace = GracefulKiller()

if len(sys.argv) != 2:
    print("Error incorrect number of arguments: " + str(sys.argv))
else:
    totalSeconds = int(sys.argv[1])

#totalSeconds = 30
toolbar_width = 70
barPerSecs = toolbar_width / totalSeconds
barsAccumulated = 0
secondsPassed = 0

# setup toolbar
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

while not grace.kill_now and secondsPassed < totalSeconds:
    rospy.sleep(1) # do real work here
    secondsPassed += 1
    bars = int(barPerSecs * secondsPassed)
    while bars > barsAccumulated:
        sys.stdout.write("#")
        sys.stdout.flush()
        barsAccumulated += 1

sys.stdout.write("\n")
