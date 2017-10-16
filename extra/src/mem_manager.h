// @file mem_manager.h
//
//  \date Created on: Oct 14, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#ifndef EXTRA_MEM_MANAGER_H_
#define EXTRA_MEM_MANAGER_H_
#include <stddef.h>

void *dn_malloc(size_t size);
void *dn_calloc(size_t n, size_t size);
void *dn_realloc(void *ptr, size_t size);
void dn_free(void *ptr);

#endif  // EXTRA_MEM_MANAGER_H_
