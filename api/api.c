#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "api.h"

image ipl_to_image(IplImage* src);

static image g_imgBuf;
static image g_imgBufLetter;
static int g_nClass = 0;
static char **g_ppName = NULL;
static network *g_pNet = NULL;
static int g_netSize = 0;
static float **g_ppPred = NULL;
static int g_iPred = 0;
static int g_nPredAvr = 3;
static float *g_pAvr = NULL;

bool yoloInit(const char *pCfgFile,
              const char *pWeightFile,
              const char *pLabelFile,
              int nPredAvr,
              int nBatch)
{
    FILE *pF = fopen(pLabelFile, "r");
    if (!pF)
        return false;
    g_nClass = 0;
    while (fgetl(pF))
        g_nClass++;
    fclose(pF);
    if (g_nClass <= 0)
        return false;

    g_ppName = get_labels(pLabelFile);
    g_pNet = load_network(pCfgFile, pWeightFile, 0);
    set_batch_network(g_pNet, nBatch);

    g_netSize = size_network(g_pNet);
    g_nPredAvr = nPredAvr;
    g_ppPred = calloc(g_nPredAvr, sizeof(float *));
    for (int i = 0; i < g_nPredAvr; i++)
    {
        g_ppPred[i] = (float *)calloc(g_netSize, sizeof(float));
    }
    g_pAvr = (float *)calloc(g_netSize, sizeof(float));

    g_imgBuf.data = NULL;

    return true;
}

int yoloUpdate(IplImage *pImg,
               yolo_object *pObj,
               int nObj,
               float thresh,
               float hier,
               float nms)
{
    if (!pImg)
        return -1;
    if (!pObj)
        return -1;
    if (nObj <= 0)
        return -1;

    if (g_imgBuf.data)
    {
        //ipl_into_image(pImg, g_imgBuf);
        g_imgBuf = ipl_to_image(pImg);
        rgbgr_image(g_imgBuf);
        letterbox_image_into(g_imgBuf, g_pNet->w, g_pNet->h, g_imgBufLetter);
    }
    else
    {
        g_imgBuf = ipl_to_image(pImg);
        rgbgr_image(g_imgBuf);
        g_imgBufLetter = letterbox_image(g_imgBuf, g_pNet->w, g_pNet->h);
    }

    layer L = g_pNet->layers[g_pNet->n - 1];
    network_predict(g_pNet, g_imgBufLetter.data);

    int i, j;
    int count;

    for (i = 0, count = 0; i < g_pNet->n; i++)
    {
        layer l = g_pNet->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            memcpy(g_ppPred[g_iPred] + count, l.output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }

    fill_cpu(g_netSize, 0, g_pAvr, 1);

    for (i = 0; i < g_nPredAvr; i++)
    {
        axpy_cpu(g_netSize, 1.0 / g_nPredAvr, g_ppPred[i], 1, g_pAvr, 1);
    }

    for (i = 0, count = 0; i < g_pNet->n; i++)
    {
        layer l = g_pNet->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            memcpy(l.output, g_pAvr + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }

    if (++g_iPred >= g_nPredAvr)
        g_iPred = 0;

    int nBox = 0;
    detection *pDet = get_network_boxes(g_pNet, g_imgBuf.w, g_imgBuf.h, thresh, hier, 0, 1, &nBox);

    if (nms > 0)
        do_nms_obj(pDet, nBox, L.classes, nms);

    int iBox = 0;
    for (i = 0; i < nBox; i++)
    {
        yolo_object *pO = &pObj[iBox];
        pO->m_topClass = -1;
        pO->m_mClass = 0;
        pO->m_topProb = 0;

        for (j = 0; j < g_nClass; j++)
        {
            float prob = pDet[i].prob[j];
            if (prob < thresh)
                continue;

            pO->m_mClass |= 1 << j;
            if (prob > pO->m_topProb)
            {
                pO->m_topClass = j;
                pO->m_topProb = prob;
            }
        }

        if (pO->m_topClass < 0)
            continue;

        box b = pDet[i].bbox;
        b.w *= 0.5;
        b.h *= 0.5;
        pO->m_l = b.x - b.w;
        pO->m_r = b.x + b.w;
        pO->m_t = b.y - b.h;
        pO->m_b = b.y + b.h;

        if (++iBox >= nObj)
            break;
    }

    free_detections(pDet, nBox);

    return iBox;
}

int yoloNClass(void)
{
    return g_nClass;
}

char *yoloGetClassName(int iClass)
{
    if (iClass < 0 || iClass >= g_nClass)
        return NULL;
    return g_ppName[iClass];
}
