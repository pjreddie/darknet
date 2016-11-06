import os
import string
import pipes

font = 'futura-normal'

def make_labels(s):
    l = string.printable
    for word in l:
        if word == ' ':
            os.system('convert -fill black -background white -bordercolor white -font %s -pointsize %d label:"\ " 32_%d.png'%(font,s,s/12-1))
        if word == '@':
            os.system('convert -fill black -background white -bordercolor white -font %s -pointsize %d label:"\@" 64_%d.png'%(font,s,s/12-1))
        elif word == '\\':
            os.system('convert -fill black -background white -bordercolor white -font %s -pointsize %d label:"\\\\\\\\" 92_%d.png'%(font,s,s/12-1))
        elif ord(word) in [9,10,11,12,13,14]:
            pass
        else:
            os.system("convert -fill black -background white -bordercolor white -font %s -pointsize %d label:%s \"%d_%d.png\""%(font,s,pipes.quote(word), ord(word),s/12-1))

for i in [12,24,36,48,60,72,84,96]:
    make_labels(i)

