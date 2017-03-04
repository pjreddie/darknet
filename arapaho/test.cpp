/*************************************************************************
 * arapaho                                                               *
 *                                                                       *
 * C++ API for Yolo v2                                                   *
 *                                                                       *
 * https://github.com/prabindh/darknet                                   *
 *                                                                       *
 * Forked from, https://github.com/pjreddie/darknet                      *
 *                                                                       *
 *   Test wrapper for arapaho                                            *
 *                                                                       *
 * Refer below file for build instructions (temporary)                   *
 *                                                                       *
 * arapaho_readme.txt                                                    *
 *                                                                       *
 *************************************************************************/

#include "arapaho.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <sys/types.h>
#include <sys/stat.h>

using namespace cv;

//
// Some configuration inputs
//
static char INPUT_DATA_FILE[]    = "input.data"; 
static char INPUT_CFG_FILE[]     = "input.cfg";
static char INPUT_WEIGHTS_FILE[] = "input.weights";
static char INPUT_IMAGE_FILE[]   = "input.jpg";

//
// Some utility functions
// 
bool fileExists(const char *file) 
{
    struct stat st;
    if(!file) return false;
    int result = stat(file, &st);
    return (0 == result);
}

//
// Main test wrapper for arapaho
//
int main()
{
    bool ret = false;
    int expectedW = 0, expectedH = 0;
    box* boxes = 0;
    
    // Early exits
    if(!fileExists(INPUT_DATA_FILE) || !fileExists(INPUT_CFG_FILE) || !fileExists(INPUT_WEIGHTS_FILE))
    {
        printf("Setup failed as inputs files do not exist or not readable!\n");
        return -1;       
    }
    
    // Create arapaho
    ArapahoV2* p = new ArapahoV2();
    if(!p)
    {
        return -1;
    }
    
    // TODO - read from arapaho.cfg    
    ArapahoV2Params ap;
    ap.datacfg = INPUT_DATA_FILE;
    ap.cfgfile = INPUT_CFG_FILE;
    ap.weightfile = INPUT_WEIGHTS_FILE;
    ap.nms = 0.4;
    ap.maxClasses = 2;
    
    // Always setup before detect
    ret = p->Setup(ap, expectedW, expectedH);
    if(false == ret)
    {
        printf("Setup failed!\n");
        if(p) delete p;
        p = 0;
        return -1;
    }
    
    // Steps below this, can be performed in a loop
    
    // loop 
    // {
    //    setup arapahoImage;
    //    p->Detect(arapahoImage);
    //    p->GetBoxes;
    // }
    //
    
    // Setup image buffer here
    ArapahoV2ImageBuff arapahoImage;
    Mat image;
    image = imread(INPUT_IMAGE_FILE, IMREAD_COLOR);

    if( image.empty() ) 
    {
        printf("Could not load the image\n");
        if(p) delete p;
        p = 0;
        return -1;
    }
    else
    {
        // Process the image
        printf("Image data = %p, w = %d, h = %d\n", image.data, image.size().width, image.size().height);
        arapahoImage.bgr = image.data;
        arapahoImage.w = image.size().width;
        arapahoImage.h = image.size().height;
        arapahoImage.channels = 3;
        // Using expectedW/H, can optimise scaling using HW in platforms where available
        
        int numObjects = 0;
        
        // Detect the objects in the image
        p->Detect(
            arapahoImage,
            0.24,
            0.5,
            numObjects);
        printf("Detected %d objects\n", numObjects);
        
        if(numObjects > 0)
        {    
            boxes = new box[numObjects];
            if(!boxes)
            {
                if(p) delete p;
                p = 0;
                return -1;
            }
            p->GetBoxes(
                boxes,
                numObjects);
        }
    }
    
clean_exit:    

    // Clear up things before exiting
    if(boxes) delete[] boxes;
    if(p) delete p;
    printf("Exiting...\n");
    return 0;
}       