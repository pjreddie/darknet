#include "arapaho.hpp"

int main()
{
    bool ret = false;
    int expectedW = 0, expectedH = 0;
    
    ArapahoV2* p = new ArapahoV2();
    if(!p)
    {
        return -1;
    }
    
    ArapahoV2Params ap;
    // read from arapaho.cfg
    ap.datacfg = "input.data";
    ap.cfgfile = "input.cfg";
    ap.weightfile = "input.weights";
    ap.nms = 0.4;
    
    p->Setup(ap, expectedW, expectedH);
    
    //Setup image buffer here
    ArapahoV2ImageBuff arapahoImage;
    Mat image;
    image = imread("input.jpg", IMREAD_COLOR);

    if( image.empty() ) 
    {
        printf("Could not load the image\n");
        if(p) delete p;
        p = 0;
        return -1;
    }
    arapahoImage.bgr = image.data;
    arapahoImage.w = image.size().width;
    arapahoImage.h = image.size().height;
    arapahoImage.channels = 3;
    // Using expectedW/H, can optimise scaling using HW in platforms where available
    
    int numObjects = 0;
    
    p->Detect(
        arapahoImage,
        0.24,
        0.5,
        200,
        numObjects);
    printf("Detected %d objects\n", numObjects);
    
    box* boxes = new box[numObjects];
    if(!boxes)
    {
        if(p) delete p;
        p = 0;
        return -1;
    }
    p->GetBoxes(
        boxes,
        numObjects);

    
    delete[] boxes;
    if(p) delete p;

    return 0;
}       