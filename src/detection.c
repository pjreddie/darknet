#include "image.h"
#include "utils.h"
#include "blas.h"
#include "cuda.h"
#include "detection.h"
#include <stdio.h>
#include <math.h>

detection* get_detections(int num, float thresh, box *boxes, float **probs, char **names, int classes)
{
	detection* dcts = (detection*)calloc(50, sizeof(detection));
	int i;
    int aux = 0;
    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            box b = boxes[i];
            dcts[aux].b = b;
            dcts[aux].classname = names[class];
            //printf("%s \n",dcts[aux].classname);
            dcts[aux].classindex = class;
            dcts[aux].prob = prob;
        	aux++;
     }

    }
    free(boxes);
    free_ptrs((void **)probs, num);
    return dcts;

}

void free_detections(detection* dec){
	free(dec);
}
