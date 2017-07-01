#ifndef __THREADS__H___

typedef void (*ThreadsCB_t)(int loop_start, int loop_end, void*data);

void threads_split(int loops, ThreadsCB_t callback, void *data);

#endif //__THREADS__H___
