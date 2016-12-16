#!/bin/sh

set -e
cat - > /tmp/image
/darknet/darknet $@ /tmp/image 2>/dev/null | grep ': [0-9]'
# /darknet/darknet classifier predict cfg/imagenet22k.dataset cfg/extraction.cfg /extraction.weights /tmp/image 2>/dev/null | grep ': [0-9]'
