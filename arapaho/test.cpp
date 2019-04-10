/*************************************************************************
 * arapaho                                                               *
 *                                                                       *
 * C++ API for Yolo v2                                                   *
 *                                                                       *
 * This test wrapper reads an image or video file and displays           *
 * detected regions in it.                                               *
 *                                                                       *
 * https://github.com/prabindh/darknet                                   *
 *                                                                       *
 * Forked from, https://github.com/pjreddie/darknet                      *
 *                                                                       *
 * Refer below file for build instructions                               *
 *                                                                       *
 * arapaho_readme.txt                                                    *
 *                                                                       *
 *************************************************************************/

#include "arapaho.hpp"
#include <string>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>

// Use OpenCV for scaling the image (faster)
#define _ENABLE_OPENCV_SCALING

using namespace cv;

//
// Some configuration inputs
//
static char INPUT_DATA_FILE[]    = "input.data"; 
static char INPUT_CFG_FILE[]     = "input.cfg";
static char INPUT_WEIGHTS_FILE[] = "input.weights";
static char INPUT_IMAGE_FILE[] = "input.jpg"; //// Can take in Image file
#define MAX_OBJECTS_PER_FRAME (100)

#define TARGET_SHOW_FPS (10)

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
    float* probs = 0;
    std::string* labels;
    
    // Early exits
    if(!fileExists(INPUT_DATA_FILE) || !fileExists(INPUT_CFG_FILE) || !fileExists(INPUT_WEIGHTS_FILE))
    {
        EPRINTF("Setup failed as input files do not exist or not readable!\n");
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
        EPRINTF("Setup failed!\n");
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

    // Setup show window
    namedWindow ( "Arapaho" , CV_WINDOW_AUTOSIZE );
    
	image = cv::imread(INPUT_IMAGE_FILE);

    // Detection step, only for image
    {
        int imageWidthPixels = 0, imageHeightPixels = 0;
        if( image.empty() ) 
        {
            EPRINTF("image.empty error - file %s\n", INPUT_IMAGE_FILE);
            if(p) delete p;
            p = 0;
            waitKey();
            return 0;
        }
        else
        {
            imageWidthPixels = image.size().width;
            imageHeightPixels = image.size().height;
            DPRINTF("Image data = %p, w = %d, h = %d\n", image.data, imageWidthPixels, imageHeightPixels);
            
            // Remember the time
            auto detectionStartTime = std::chrono::system_clock::now();

            // Process the image
            arapahoImage.bgr = image.data;
            arapahoImage.w = imageWidthPixels;
            arapahoImage.h = imageHeightPixels;
            arapahoImage.channels = 3;
            // Using expectedW/H, can optimise scaling using HW in platforms where available
            
            int numObjects = 0;
            
#ifdef _ENABLE_OPENCV_SCALING
            // Detect the objects in the image
            p->Detect(
                image,
                0.24,
                0.5,
                numObjects);
#else
            p->Detect(
                arapahoImage,
                0.24,
                0.5,
                numObjects);
#endif
            std::chrono::duration<double> detectionTime = (std::chrono::system_clock::now() - detectionStartTime);
            
            printf("==> Detected [%d] objects in [%f] seconds\n", numObjects, detectionTime.count());
            
            if(numObjects > 0 && numObjects < MAX_OBJECTS_PER_FRAME) // Realistic maximum
            {    
                boxes = new box[numObjects];
                probs = new float[numObjects];
                labels = new std::string[numObjects];
                if(!boxes)
                {
                    if(p) delete p;
                    p = 0;
                    return -1;
                }
                if(!labels)
                {
                    if(p) delete p;
                    p = 0;
                    if(boxes)
                    {
                        delete[] boxes;
                        boxes = NULL;                        
                    }
                    if (probs)
                    {
                        delete[] probs;
                        probs = NULL;
                    }
                    return -1;
                }
                
                // Get boxes and labels
                p->GetBoxes(
                    boxes,
                    labels,
                    probs,
                    numObjects
                    );
                
                int objId = 0;
                int leftTopX = 0, leftTopY = 0, rightBotX = 0,rightBotY = 0;
                for (objId = 0; objId < numObjects; objId++)
                {
                    leftTopX = 1 + imageWidthPixels*(boxes[objId].x - boxes[objId].w / 2);
                    leftTopY = 1 + imageHeightPixels*(boxes[objId].y - boxes[objId].h / 2);
                    rightBotX = 1 + imageWidthPixels*(boxes[objId].x + boxes[objId].w / 2);
                    rightBotY = 1 + imageHeightPixels*(boxes[objId].y + boxes[objId].h / 2);
                    DPRINTF("Box #%d: center {x,y}, box {w,h} = [%f, %f, %f, %f]\n", 
                            objId, boxes[objId].x, boxes[objId].y, boxes[objId].w, boxes[objId].h);
                    // Show image and overlay using OpenCV
                    rectangle(image,
                        cvPoint(leftTopX, leftTopY),
                        cvPoint(rightBotX, rightBotY),
                        CV_RGB(255, 0, 0), 1, 8, 0);
                    // Show labels
                    if (labels[objId].c_str())
                    {
                        char probAsString[200];
                        snprintf(probAsString, 200, "%s: %f", labels[objId].c_str(), probs[objId]);
                        DPRINTF("Label:%s, Probabilty: %f\n\n", labels[objId].c_str(), probAsString);
                        putText(image, probAsString, cvPoint(leftTopX, leftTopY),
                            FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
                    }
                }
                
                if (boxes)
                {
                    delete[] boxes;
                    boxes = NULL;
                }
                if (labels)
                {
                    delete[] labels;
                    labels = NULL;
                }   
                if (probs)
                {
                    delete[] probs;
                    probs = NULL;
                }
                
            }// If objects were detected
            imshow("Arapaho", image);
            waitKey((1000 / TARGET_SHOW_FPS));
         
        } //If a frame was read
    }// Detection loop
    
clean_exit:    

    // Clear up things before exiting
    if(p) delete p;
    DPRINTF("Exiting...\n");
    return 0;
}       
