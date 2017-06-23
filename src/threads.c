#define _GNU_SOURCE

#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include "threads.h"

static int stick_this_thread_to_core(int core_id) {
  int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (core_id < 0 || core_id >= num_cores)
  return EINVAL;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  pthread_t current_thread = pthread_self();    
  return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}


typedef struct {
  int loops;
  int loop_start;
  int loop_end;
  int core_id;
  void * data; //customer data
  ThreadsCB_t callback;
} ThreadInfo_t; 

void thread_main(void *d) {
  ThreadInfo_t *thread_input = (ThreadInfo_t*)d;
  int ret = stick_this_thread_to_core(thread_input->core_id);
  if (ret!=0) {
      fprintf(stderr, "Error to stick thread %d\n", thread_input->core_id);
      exit(1);
  }
  thread_input->callback (thread_input->loop_start, thread_input->loop_end, thread_input->data);
}

void threads_split (int loops, ThreadsCB_t callback, void *data) {
  int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  pthread_t thread_ptr[128];
  ThreadInfo_t thread_input[128] = {{0}};

  for (int i=0; i < num_cores; i++) {
    /* create a second thread which executes inc_x(&x) */
    thread_input[i].loops      = loops; 
    thread_input[i].callback   = callback; 
    thread_input[i].data       = data; 
    thread_input[i].loop_start = i*loops/num_cores;
    int end = (i+1)*loops/num_cores;
    thread_input[i].loop_end   = end<loops?end:loops;
    thread_input[i].core_id    = i;
    if(pthread_create(&(thread_ptr[i]), NULL, thread_main, &(thread_input[i]))) {
      fprintf(stderr, "Error creating thread\n");
      exit(1);
    }
    //printf ("created thread: %d\n", i);
  }

  /* wait for the threads to finish */
  for (int i=0; i < num_cores; i++) {
    if(pthread_join(thread_ptr[i], NULL)) {
      fprintf(stderr, "Error joining thread %d\n", i);
      exit(2);
    }
  }
}

