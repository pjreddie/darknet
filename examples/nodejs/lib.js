lib = require("node-dnet")
fs = require("fs")

network = new lib.Network("/var/trains/generico/net.weights","/var/trains/generico/net.cfg","/var/trains/generico/net.names")

im = lib.loadImageBuffer(fs.readFileSync('dog.jpg'))
p = network.predict(im)
console.log(lib.getpredictdata(p))
lib.drawDetecions(im,p)
console.log(lib.saveImage(im,"out2"))
