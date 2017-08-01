#include "image.h"
#include "utils.h"
#include "blas.h"
#include "cuda.h"
#include "detection.h"
#include <stdio.h>
#include <math.h>

int get_detections(int num, float thresh, box *boxes, float **probs, char **names, int classes,detection* dcts)
{
	int count = 0;
	int i;
    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            box b = boxes[i];
            dcts[count].b = b;
            dcts[count].classname = names[class];
            //printf("%s \n",dcts[aux].classname);
            dcts[count].classindex = class;
            dcts[count].prob = prob;
        	count++;
     }

    }
    free(boxes);
    free_ptrs((void **)probs, num);
    return count;
}

void free_detections(detection* dec){
	free(dec);
}
