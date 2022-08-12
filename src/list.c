#include <stdlib.h>
#include <string.h>
#include "list.h"

dn_list *make_list()
{
	dn_list *l = malloc(sizeof(dn_list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(dn_list *l){
    if(!l->back) return 0;
    dn_node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}

void list_insert(dn_list *l, void *val)
{
	dn_node *new = malloc(sizeof(dn_node));
	new->val = val;
	new->next = 0;

	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		l->back->next = new;
		new->prev = l->back;
	}
	l->back = new;
	++l->size;
}

void free_node(dn_node *n)
{
	dn_node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(dn_list *l)
{
	free_node(l->front);
	free(l);
}

void free_list_contents(dn_list *l)
{
	dn_node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

void **list_to_array(dn_list *l)
{
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    dn_node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
