from sys import argv

with open(argv[1],'r+') as iFile:
    names = filter(lambda e: len(e) > 0, sorted(iFile.read().split("\n")))
    iFile.seek(0,0)
    iFile.write("\n".join(names))    
