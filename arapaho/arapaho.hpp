/*
 * 
 * arapaho - API wrapper for darknet
 * 
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "region_layer.h"
#include "option_list.h"

struct ArapahoV2Params
{
    char* datacfg;
    char* cfgfile;
    char* weightfile;
    float nms;
};
struct ArapahoV2ImageBuff
{
    float* bgr;
    int w;
    int h;
};

class ArapahoV2
{
public:
    ArapahoV2();    
    ~ArapahoV2();
    
    bool Setup(ArapahoV2Params & p);
    bool Detect(
            ArapahoV2ImageBuff & imageBuff, 
            float thresh, 
            float hier_thresh,
            int   maxClasses,            
            int & objectCount);
    bool GetBoxes(box* outBoxes, int boxCount);
private:    
    box     *boxes;
    float   **probs;
    network net;
    layer   l;
    float   nms;
    int     maxClasses;
    int     threshold;
};