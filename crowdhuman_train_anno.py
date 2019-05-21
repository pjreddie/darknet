import numpy
import cv2

import yaml

intPERSON = 0
intHEAD   = 1

with open('annotation_train.odgt') as f:
    for line in f:
        
        dictLine = yaml.load(line)

        strID = dictLine['ID']

        img = cv2.imread('crowdhuman_train/{}.jpg'.format(strID),1)

        imgWidth  = img.shape[1]
        imgHeight = img.shape[0]

        # Create .txt label file
        with open('crowdhuman_train/{}.txt'.format(strID), 'w+') as txtf:
            
            for label in dictLine['gtboxes']:

                # Person BB
                px = float(label['fbox'][0])
                py = float(label['fbox'][1])
                pw = float(label['fbox'][2])
                ph = float(label['fbox'][3])

                # Head BB
                hx = float(label['hbox'][0])
                hy = float(label['hbox'][1])
                hw = float(label['hbox'][2])
                hh = float(label['hbox'][3])

                # Absolute person BB
                cpx = px + pw/2
                cpy = py + ph/2

                abspx = cpx / imgWidth
                abspy = cpy / imgHeight
                abspw = pw / imgWidth
                absph = ph / imgHeight  
                

                # Absolute head BB
                chx = hx + hw/2
                chy = hy + hh/2

                abshx = chx / imgWidth
                abshy = chy / imgHeight
                abshw = hw / imgWidth
                abshh = hh / imgHeight  

                # Write to file
                txtf.write('{} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(intPERSON,
                                                                     abspx,
                                                                     abspy,
                                                                     abspw,
                                                                     absph))

                txtf.write('{} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(intHEAD,
                                                                     abshx,
                                                                     abshy,
                                                                     abshw,
                                                                     abshh))
        

