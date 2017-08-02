typedef struct{
    box b;
    int classindex;
    char* classname;
    float prob;
} detection;


int get_detections(int num, float thresh, box *boxes, float **probs, char **names, int classes, detection * dcts);
