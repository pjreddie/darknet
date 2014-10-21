#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

float sec(clock_t clocks)
{
    return (float)clocks/CLOCKS_PER_SEC;
}

void error(char *s)
{
    fprintf(stderr, "Error: %s\n", s);
    exit(0);
}

void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

list *split_str(char *s, char delim)
{
    int i;
    int len = strlen(s);
    list *l = make_list();
    list_insert(l, s);
    for(i = 0; i < len; ++i){
        if(s[i] == delim){
            s[i] = '\0';
            list_insert(l, &(s[i+1]));
        }
    }
    return l;
}

void strip(char *s)
{
    int i;
    int len = strlen(s);
    int offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

void strip_char(char *s, char bad)
{
    int i;
    int len = strlen(s);
    int offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==bad) ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    int size = 512;
    char *line = malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    int curr = strlen(line);
    
    while(line[curr-1]!='\n'){
        size *= 2;
        line = realloc(line, size*sizeof(char));
        if(!line) malloc_error();
        fgets(&line[curr], size-curr, fp);
        curr = strlen(line);
    }
    line[curr-1] = '\0';

    return line;
}

char *copy_string(char *s)
{
    char *copy = malloc(strlen(s)+1);
    strncpy(copy, s, strlen(s)+1);
    return copy;
}

list *parse_csv_line(char *line)
{
    list *l = make_list();
    char *c, *p;
    int in = 0;
    for(c = line, p = line; *c != '\0'; ++c){
        if(*c == '"') in = !in;
        else if(*c == ',' && !in){
            *c = '\0';
            list_insert(l, copy_string(p));
            p = c+1;
        }
    }
    list_insert(l, copy_string(p));
    return l;
}

int count_fields(char *line)
{
	int count = 0;
	int done = 0;
    char *c;
	for(c = line; !done; ++c){
		done = (*c == '\0');
		if(*c == ',' || done) ++count;
	}
	return count;
}

float *parse_fields(char *line, int n)
{
	float *field = calloc(n, sizeof(float));
	char *c, *p, *end;
	int count = 0;
	int done = 0;
	for(c = line, p = line; !done; ++c){
		done = (*c == '\0');
		if(*c == ',' || done){
			*c = '\0';
			field[count] = strtod(p, &end);
			if(p == c) field[count] = nan("");
			if(end != c && (end != c-1 || *end != '\r')) field[count] = nan(""); //DOS file formats!
			p = c+1;
			++count;
		}
	}
	return field;
}

float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}

float mean_array(float *a, int n)
{
    return sum_array(a,n)/n;
}

float variance_array(float *a, int n)
{
    int i;
    float sum = 0;
    float mean = mean_array(a, n);
    for(i = 0; i < n; ++i) sum += (a[i] - mean)*(a[i]-mean);
    float variance = sum/n;
    return variance;
}

float constrain(float a, float max)
{
    if(a > abs(max)) return abs(max);
    if(a < -abs(max)) return -abs(max);
    return a;
}

void normalize_array(float *a, int n)
{
    int i;
    float mu = mean_array(a,n);
    float sigma = sqrt(variance_array(a,n));
    for(i = 0; i < n; ++i){
        a[i] = (a[i] - mu)/sigma;
    }
    mu = mean_array(a,n);
    sigma = sqrt(variance_array(a,n));
}

void translate_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i){
        a[i] += s;
    }
}

void scale_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i){
        a[i] *= s;
    }
}
int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

float rand_normal()
{
    int i;
    float sum= 0;
    for(i = 0; i < 12; ++i) sum += (float)rand()/RAND_MAX;
    return sum-6.;
}
float rand_uniform()
{
    return (float)rand()/RAND_MAX;
}

float **one_hot_encode(float *a, int n, int k)
{
    int i;
    float **t = calloc(n, sizeof(float*));
    for(i = 0; i < n; ++i){
        t[i] = calloc(k, sizeof(float));
        int index = (int)a[i];
        t[i][index] = 1;
    }
    return t;
}

