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

#include "arapaho.hpp"

ArapahoV2::ArapahoV2()
{
        boxes = 0;
        probs = 0;    
        l = {};
        net = {};
        maxClasses = 0;
        threshold = 0;
        bSetup = false;
}
    
ArapahoV2::~ArapahoV2()
{
    // TODO - Massive cleanup here
    
    if(boxes) 
        free(boxes);
    if(probs)
        free_ptrs((void **)probs, l.w*l.h*l.n);
    boxes = 0;
    probs = 0;
    bSetup = false;
}
    
bool ArapahoV2::Setup(
            ArapahoV2Params & p,
            int & expectedWidth,
            int & expectedHeight
                     )
{
    expectedHeight = expectedWidth = 0;
    
    // TODO
#if 0    
    if(!p.datacfg)
    {
        DPRINTF("No data configuration file specified!\n");
        return false;
    }    
    
    list *options = read_data_cfg(p.datacfg);
    char *name_list = option_find_str(options, "names", 
                            "data/names.list");
    char **names = get_labels(name_list);
#endif    
    int j;
    bool ret = false;
    
    bSetup = false;
    
    // Early exits
    if(!p.cfgfile)
    {
        DPRINTF("No cfg file specified!\n");
        return false;
    }

    if(!p.weightfile)
    {
        DPRINTF("No weights file specified!\n");
        return false;
    }    

    // Print some debug info
    nms = p.nms;
    maxClasses = p.maxClasses;
    
    net = parse_network_cfg(p.cfgfile);
    DPRINTF("Setup: net.n = %d\n", net.n);   
    DPRINTF("net.layers[0].batch = %d\n", net.layers[0].batch);
    
    load_weights(&net, p.weightfile);
    set_batch_network(&net, 1);     
    l = net.layers[net.n-1];
    DPRINTF("Setup: layers = %d, %d, %d\n", l.w, l.h, l.n);

    // Class limiter
    if(l.classes > maxClasses)
    {
        DPRINTF("Warning: Read classes from cfg (%d) > maxClasses (%d)", l.classes, maxClasses);
    }    
    
    expectedHeight = net.h;
    expectedWidth = net.w;
    DPRINTF("Image expected w,h = [%d][%d]!\n", net.w, net.h);            
    
    boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
    
    // Error exits
    if(!boxes || !probs)
    {
        DPRINTF("Error allocating boxes/probs, %p/%p !\n", boxes, probs);
        goto clean_exit;
    }

    for(j = 0; j < l.w*l.h*l.n; ++j) 
    {
        probs[j] = (float*)calloc(l.classes + 1, sizeof(float));
        if(!probs[j])
        {
            DPRINTF("Error allocating probs[%d]!\n", j);            
            goto clean_exit;
        }
    }
    ret = true;
    bSetup = ret;
    DPRINTF("Setup: Done\n");
    return ret;
    
clean_exit:        
    if(boxes) 
        free(boxes);
    if(probs)
        free_ptrs((void **)probs, l.w*l.h*l.n);
    
    return ret;
}

//
// Detect API to get objects detected
// \warning Setup must have been done on the object before
//
bool ArapahoV2::Detect(
            ArapahoV2ImageBuff & imageBuff, 
            float thresh, 
            float hier_thresh,
            int & objectCount)
{
    int i, j, k, count=0;
        
    objectCount = 0;
    threshold = thresh;
    
    // Early exits
    if(!bSetup)
    {
        DPRINTF("Not Setup!\n");
        return false;
    }
    if(!imageBuff.bgr)
    {
        DPRINTF("Error in imageBuff! [bgr = %d, w = %d, h = %d]\n",
                    !imageBuff.bgr, imageBuff.w != net.w, 
                        imageBuff.h != net.h);
        return false;        
    }        
    
    // Get the image to suit darknet
    image inputImage = make_image(imageBuff.w, 
                                imageBuff.h, imageBuff.channels);
    if(!inputImage.data || inputImage.w != imageBuff.w || 
            inputImage.h != imageBuff.h)
    {
        DPRINTF("Error in inputImage! [bgr = %d, w = %d, h = %d]\n", 
                    !inputImage.data, inputImage.w != net.w, 
                        inputImage.h != net.h);
        return false;        
    }     
    // Convert the bytes to float
    int step = imageBuff.channels*imageBuff.w;
    for(k= 0; k < imageBuff.channels; ++k){
        for(i = 0; i < imageBuff.h; ++i){
            for(j = 0; j < imageBuff.w; ++j){
                int offset = i*step + j*imageBuff.channels + k;
                inputImage.data[count++] =                     
                    (float)imageBuff.bgr[offset]/(float)255.;
            }
        }
    }
    inputImage.h = imageBuff.h;
    inputImage.w = imageBuff.w;
    inputImage.c = imageBuff.channels;
    
    if (inputImage.h != net.h || inputImage.w != net.w)
    {
        DPRINTF("Detect: Resizing image to match network \n");
        inputImage = resize_image(inputImage, net.w, net.h);
        if(!inputImage.data || inputImage.w != net.w || 
                inputImage.h != net.h)
        {
            DPRINTF("Error in resized img! [data = %d, w = %d, h = %d]\n",
                        !inputImage.data, inputImage.w != net.w, 
                            inputImage.h != net.h);
            return false;        
        }         
    }
    // Onto safe lands from here
    // Predict
    network_predict(net, inputImage.data);
    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
    
    DPRINTF("l.softmax_tree = %p, nms = %f\n", l.softmax_tree, nms);
    if (l.softmax_tree && nms)
    {
        do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    }
    else if (nms) 
        do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    
    // Update object counts
    for(i = 0; i < (l.w*l.h*l.n); ++i){
        int class1 = max_index(probs[i], l.classes);
        float prob = probs[i][class1];
        if(prob > thresh){
            objectCount ++;
        }
    }

    return true;
}
    
//
// Query API to get box coordinates for objects detected
//
bool ArapahoV2::GetBoxes(box* outBoxes, int boxCount)
{
    
    int count = 0;
    int i;
    
    if(!boxes || !probs)
    {
        DPRINTF("Error NULL boxes/probs, %p/%p !\n", boxes, probs);
        return false;
    }
    for(i = 0; i < (l.w*l.h*l.n); ++i)
    {
        int class1 = max_index(probs[i], l.classes);
        float prob = probs[i][class1];
        if(prob > threshold && count < boxCount)
        {
            outBoxes[count ++] = boxes[i];
        }
    }
    
    return true;
}
