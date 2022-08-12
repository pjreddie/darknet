#ifndef LIST_H
#define LIST_H
#include "darknet.h"

dn_list *make_list();
int list_find(dn_list *l, void *val);

void list_insert(dn_list *, void *);


void free_list_contents(dn_list *l);

#endif
