#ifndef TABLE_H
#define TABLE_H
#include "darknet.h"

table copy_table(table t);

void table_to_csv(table t1, table t2, char *filename);
void print_table(table t);

#endif
