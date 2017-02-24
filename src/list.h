#ifndef LIST_H
#define LIST_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dn_node{
    void *val;
    struct dn_node *next;
    struct dn_node *prev;
} dn_node;

typedef struct list{
    int size;
    dn_node *front;
    dn_node *back;
} list;

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);

void **list_to_array(list *l);

void free_list(list *l);
void free_list_contents(list *l);

#ifdef __cplusplus
}
#endif

#endif
