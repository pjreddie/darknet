#!/bin/bash
./darknet rnn generatetactic cfg/gru.cfg peek.weights < $1 2>/dev/null
