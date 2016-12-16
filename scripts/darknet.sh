#!/bin/sh

set -e
cat - > /tmp/image
/darknet/darknet $@ /tmp/image 2>/dev/null | grep ': [0-9]'
