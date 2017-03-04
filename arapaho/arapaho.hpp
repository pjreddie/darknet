/*************************************************************************
 * arapaho                                                               *
 *                                                                       *
 * C++ API for Yolo v2 (Detection)                                       *
 *                                                                       *
 * https://github.com/prabindh/darknet                                   *
 *                                                                       *
 * Forked from, https://github.com/pjreddie/darknet                      *
 *                                                                       *
 *************************************************************************/

#ifndef _ENABLE_ARAPAHO
#define _ENABLE_ARAPAHO

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

//////////////////////////////////////////////////////////////////////////

#define ARAPAHO_MAX_CLASSES (200)

#ifdef _DEBUG
#define DPRINTF printf
#else
#define DPRINTF
#endif

//////////////////////////////////////////////////////////////////////////

struct ArapahoV2Params
{
    char* datacfg;
    char* cfgfile;
    char* weightfile;
    float nms;
    int maxClasses;
};
struct ArapahoV2ImageBuff
{
    unsigned char* bgr;
    int w;
    int h;
    int channels;
};

//////////////////////////////////////////////////////////////////////////

class ArapahoV2
{
public:
    ArapahoV2();    
    ~ArapahoV2();
    
    bool Setup(ArapahoV2Params & p,
            int & expectedWidth,
            int & expectedHeight);
    
    bool Detect(
            ArapahoV2ImageBuff & imageBuff, 
            float thresh, 
            float hier_thresh,         
            int & objectCount);
    
    bool GetBoxes(box* outBoxes, int boxCount);
private:    
    box     *boxes;
    float   **probs;
    bool    bSetup;
    network net;
    layer   l;
    float   nms;
    int     maxClasses;
    int     threshold;
};

//////////////////////////////////////////////////////////////////////////

#endif // _ENABLE_ARAPAHO