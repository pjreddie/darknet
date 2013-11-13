#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include "list.h"

void error(char *s);
void malloc_error();
void file_error(char *s);
void strip(char *s);
void strip_char(char *s, char bad);
list *split_str(char *s, char delim);
char *fgetl(FILE *fp);
list *parse_csv_line(char *line);
char *copy_string(char *s);
int count_fields(char *line);
double *parse_fields(char *line, int n);
#endif

