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
    classNames = 0;
    l = {};
    net = {};
    nms = 0;
    maxClasses = 0;
    threshold = 0;
    bSetup = false;
    setlocale(LC_NUMERIC,"C");
}
    
ArapahoV2::~ArapahoV2()
{
    // TODO - Massive cleanup here
    
    if(boxes) 
        free(boxes);
    if(probs)
        free_ptrs((void **)probs, l.w*l.h*l.n);
    if(classNames)
    {
        //todo
    }
    boxes = 0;
    probs = 0;
    classNames = 0;
    bSetup = false;
}
    
bool ArapahoV2::Setup(
            ArapahoV2Params & p,
            int & expectedWidth,
            int & expectedHeight
                     )
{
    expectedHeight = expectedWidth = 0;
    
    if(!p.datacfg)
    {
        DPRINTF("No data configuration file specified!\n");
        return false;
    }    
    
    list *options = read_data_cfg(p.datacfg);
    char nameField[] = "names";
    char defaultName[] = "data/names.list";
    char *nameListFile = option_find_str(options, nameField, defaultName);
    if(!nameListFile)
    {
        DPRINTF("No valid nameList file specified in options file [%s]!\n", p.datacfg);
        return false;
    }
    classNames = get_labels(nameListFile);
    if(!classNames)
    {
        DPRINTF("No valid class names specified in nameList file [%s]!\n", nameListFile);
        return false;
    }

    int j;
    bool ret = false;
    
    bSetup = false;
    
    // Early exits
    if(!p.cfgfile)
    {
        EPRINTF("No cfg file specified!\n");
        return false;
    }

    if(!p.weightfile)
    {
        EPRINTF("No weights file specified!\n");
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
        EPRINTF("Warning: Read classes from cfg (%d) > maxClasses (%d)\n", l.classes, maxClasses);
    }    
    
    expectedHeight = net.h;
    expectedWidth = net.w;
    DPRINTF("Image expected w,h = [%d][%d]!\n", net.w, net.h);            
    
    boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
    
    // Error exits
    if(!boxes || !probs)
    {
        EPRINTF("Error allocating boxes/probs, %p/%p !\n", boxes, probs);
        goto clean_exit;
    }

    for(j = 0; j < l.w*l.h*l.n; ++j) 
    {
        probs[j] = (float*)calloc(l.classes + 1, sizeof(float));
        if(!probs[j])
        {
            EPRINTF("Error allocating probs[%d]!\n", j);            
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
    boxes = NULL;
    probs = NULL;
    return ret;
}

//
// Detect API to get objects detected
// \warning Takes in raw image input as argument, converts to float and uses darknet image functions to resize. Can be slower.
// Use the CV variant for performance
// \warning Setup must have been done on the object before
//
bool ArapahoV2::Detect(
    ArapahoV2ImageBuff & imageBuff,
    float thresh,
    float hier_thresh,
    int & objectCount)
{
    int i, j, k, count = 0;

    objectCount = 0;
    threshold = thresh;

    // Early exits
    if (!bSetup)
    {
        EPRINTF("Not Setup!\n");
        return false;
    }
    if (!imageBuff.bgr)
    {
        EPRINTF("Error in imageBuff! [bgr = %d, w = %d, h = %d]\n",
            !imageBuff.bgr, imageBuff.w != net.w,
            imageBuff.h != net.h);
        return false;
    }

    // Get the image to suit darknet
    image inputImage = make_image(imageBuff.w,
        imageBuff.h, imageBuff.channels);
    if (!inputImage.data || inputImage.w != imageBuff.w ||
        inputImage.h != imageBuff.h)
    {
        EPRINTF("Error in inputImage! [bgr = %d, w = %d, h = %d]\n",
            !inputImage.data, inputImage.w != net.w,
            inputImage.h != net.h);
        free_image(inputImage);
        return false;
    }
    // Convert the bytes to float
    int step = imageBuff.channels*imageBuff.w;
    for (k = 0; k < imageBuff.channels; ++k){
        for (i = 0; i < imageBuff.h; ++i){
            for (j = 0; j < imageBuff.w; ++j){
                int offset = i*step + j*imageBuff.channels + k;
                inputImage.data[count++] =
                    (float)imageBuff.bgr[offset] / (float)255.;
            }
        }
    }
    inputImage.h = imageBuff.h;
    inputImage.w = imageBuff.w;
    inputImage.c = imageBuff.channels;

    if (inputImage.h != net.h || inputImage.w != net.w)
    {
        DPRINTF("Detect: Resizing image to match network \n");
        // Free the original buffer, and assign a new resized buffer
        image inputImageTemp = resize_image(inputImage, net.w, net.h);
        free_image(inputImage);
        inputImage = inputImageTemp;
        if (!inputImage.data || inputImage.w != net.w ||
            inputImage.h != net.h)
        {
            EPRINTF("Error in resized img! [data = %d, w = %d, h = %d]\n",
                !inputImage.data, inputImage.w != net.w,
                inputImage.h != net.h);
            return false;
        }
    }
    
    __Detect(inputImage.data, thresh, hier_thresh, objectCount);

    free_image(inputImage);
    return true;
}


//
// Detect API to get objects detected
// \warning Takes in OpenCV Mat image structure as argument
// \warning Setup must have been done on the object before
//
bool ArapahoV2::Detect(
            const cv::Mat & inputMat,
            float thresh, 
            float hier_thresh,
            int & objectCount)
{
    int i, count=0;
        
    objectCount = 0;
    threshold = thresh;
    
    // Early exits
    if(!bSetup)
    {
        EPRINTF("Not Setup!\n");
        return false;
    }
    if(inputMat.empty())
    {
        EPRINTF("Error in inputImage! [bgr = %d, w = %d, h = %d]\n",
                    !inputMat.data, inputMat.cols != net.w,
                    inputMat.rows != net.h);
        return false;
    }
    
    //Convert to rgb
    cv::Mat inputRgb;
    cvtColor(inputMat, inputRgb, CV_BGR2RGB);
    // Convert the bytes to float
    cv::Mat floatMat;
    inputRgb.convertTo(floatMat, CV_32FC3, 1/255.0);

    if (floatMat.rows != net.h || floatMat.cols != net.w)
    {
        DPRINTF("Detect: Resizing image to match network \n");
        resize(floatMat, floatMat, cv::Size(net.w, net.h));
    }
    // Get the image to suit darknet
    std::vector<cv::Mat> floatMatChannels(3);
    split(floatMat, floatMatChannels);
    vconcat(floatMatChannels, floatMat);

    __Detect((float*)floatMat.data, thresh, hier_thresh, objectCount);

    return true;
}
    
//
// Query API to get box coordinates and box labels for objects detected
//
bool ArapahoV2::GetBoxes(box* outBoxes, std::string* outLabels, int boxCount)
{
    
    int count = 0;
    int i;
    
    if(!boxes || !probs || !outLabels || !outBoxes)
    {
        EPRINTF("Error NULL boxes/probs, %p, %p !\n", boxes, probs);
        return false;
    }
    for(i = 0; i < (l.w*l.h*l.n); ++i)
    {
        int classIndex = max_index(probs[i], l.classes);
        float prob = probs[i][classIndex];
        if(prob > threshold && count < boxCount)
        {
            outLabels[count] = std::string(classNames[classIndex]);
            outBoxes[count]  = boxes[i];
            count ++;
        }
    }
    
    return true;
}

//////////////////////////////////////////////////////////////////
/// Private APIs
//////////////////////////////////////////////////////////////////
void ArapahoV2::__Detect(float* inData, float thresh, float hier_thresh, int & objectCount)
{
    int i;
    // Predict
    network_predict(net, inData);
    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);

    DPRINTF("l.softmax_tree = %p, nms = %f\n", l.softmax_tree, nms);
    if (l.softmax_tree && nms)
    {
        do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    }
    else if (nms)
        do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    // Update object counts
    for (i = 0; i < (l.w*l.h*l.n); ++i){
        int class1 = max_index(probs[i], l.classes);
        float prob = probs[i][class1];
        if (prob > thresh){
            objectCount++;
        }
    }
}
