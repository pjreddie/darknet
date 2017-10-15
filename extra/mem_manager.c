// @file mem_manager.c
//
//  \date Created on: Oct 14, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//
#include <stdlib.h>
#include <stdio.h>

// Wrapper functions to help with memory profiling.
void *dn_malloc(size_t size) {
  void *ptr = malloc(size);
  if (!ptr) {
    printf("Malloc failed\n");
  }
  return ptr;
}

void *dn_calloc(size_t n, size_t size) {
  void *ptr = calloc(n, size);
  if (!ptr) {
    printf("Calloc failed\n");
  }
  return ptr;
}

void *dn_realloc(void *ptr, size_t size) {
  ptr = realloc(ptr, size);
  if (!ptr) {
    printf("Realloc failed\n");
  }
  return ptr;
}

void dn_free(void *ptr) {
  if (ptr) {
    free(ptr);
  }
}
