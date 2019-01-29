#ifndef DARKNET_API_H
#define DARKNET_API_H
#include <stdbool.h>

#include "opencv2/highgui/highgui_c.h"

typedef struct yolo_object
{
	int m_topClass;
	float m_topProb;
	long m_mClass; //all candidate class mask
	float m_l;
	float m_t;
	float m_r;
	float m_b;
} yolo_object;

bool yoloInit(const char *pCfgFile,
			  const char *pWeightFile,
			  const char *pLabelFile,
			  int nPredAvr,
			  int nBatch);
int yoloUpdate(IplImage *pImg,
			   yolo_object *pObj,
			   int nObj,
			   float thresh,
			   float hier,
			   float nms);
int yoloNClass(void);
char *yoloGetClassName(int iClass);

#endif
