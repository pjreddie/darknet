#include <stdlib.h>
#include <string.h>
#include "list.h"

list *make_list() // make list function
{
	list *l = malloc(sizeof(list)); // list ?™?  ?• ?‹¹ 
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

void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}

void list_insert(list *l, void *val) // list_insert() function
{ // val is kvp struct type this type can find in option_list.h file.
	node *new = malloc(sizeof(node));
	new->val = val;// new->val have 3 various two char * and one int type various
	new->next = 0; //if this is first new->next is 0 but it is not first 
				   // new->next will be changed.
	if(!l->back){// if this node insert first
		l->front = new; // this is first list
		new->prev = 0; // haven't prev
	}else{ // it is now first node in list
		l->back->next = new; // next node is new
		new->prev = l->back; // new node's prev is list's back
	}
	l->back = new;
	++l->size;
}

void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(list *l)
{
	free_node(l->front);
	free(l);
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

void **list_to_array(list *l)
{
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
