#ifndef LIST_H
#define LIST_H
#include "darknet.h"

list *make_list();
node *list_find(list *, void *);

void list_insert(list *, void *);


void free_list_contents(list *l);

#endif
