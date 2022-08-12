#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


int read_option(char *s, dn_list *options);
void option_insert(dn_list *l, char *key, char *val);
char *option_find(dn_list *l, char *key);
float option_find_float(dn_list *l, char *key, float def);
float option_find_float_quiet(dn_list *l, char *key, float def);
void option_unused(dn_list *l);

#endif
