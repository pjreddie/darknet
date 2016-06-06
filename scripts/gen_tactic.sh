#!/bin/bash
# Usage:
# wget http://pjreddie.com/media/files/peek.weights
# scripts/gen_tactic.sh < data/goal.txt
./darknet rnn generatetactic cfg/gru.cfg peek.weights 2>/dev/null
