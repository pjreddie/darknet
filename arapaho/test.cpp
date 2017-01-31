#include "arapaho.hpp"

int main()
{
    bool ret = false;
    ArapahoV2* p = new ArapahoV2();
    if(!p)
    {
        return -1;
    }
    
    ArapahoV2Params ap;
    ap.datacfg = "";
    ap.cfgfile = "";
    ap.weightfile = "";
    ap.nms = 0.4;
    
    p->Setup(ap);
    
    ArapahoV2ImageBuff image;
    //Setup image buffer here
    
    int numObjects = 0;
    
    p->Detect(
        image,
        0.24,
        0.5,
        200,
        numObjects);
    
    box* boxes = new box[numObjects];
    if(!boxes)
    {
        if(p) delete p;
        return -1;
    }
    p->GetBoxes(
        boxes,
        numObjects);

    printf("Detected %d objects\n", numObjects);
    
    delete[] boxes;
    if(p) delete p;

    return 0;
}       