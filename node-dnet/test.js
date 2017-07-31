var ffi = require('ffi');
var Struct = require('ref-struct');
var ArrayType = require('ref-array');
var ref = require('ref')

var charPtr = ref.refType('char');
var StringArray = ArrayType('string');
var charPtrPtr = ref.refType(charPtr);
var CharPtrArray = ArrayType(charPtr);


var Image = Struct({
  'w': 'int',
  'h': 'int',
  'c': 'int',
  'data': 'pointer'
});

var Box = Struct({
  'x': 'float',
  'y': 'float',
  'w': 'float',
  'h': 'float'
});


var Detection = Struct({
  'box': Box,
  'classindex': 'int',
  'classname': 'string',
  'prob': 'float',
});


var DetectionArray = ArrayType(Detection,2000);


var lib = ffi.Library('libdarknet', {
  'load_network_p': [ 'pointer', [ 'string','string','int' ] ],
  'load_image_color': [ Image, [ 'string','int','int' ] ],
  'predict': [ DetectionArray, [ 'pointer',Image, 'float','string'] ],
});


net = lib.load_network_p("/var/trains/generico/net.cfg", "/var/trains/generico/net.weights",0);
im  = lib.load_image_color("dog.jpg",0,0);
a = lib.predict(net,im,0.25,"/var/trains/generico/net.names")

console.log(a[1])
