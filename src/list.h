#ifndef LIST_H
#define LIST_H
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);


void free_list_contents(list *l);

#ifdef __cplusplus
}
#endif

#endif
