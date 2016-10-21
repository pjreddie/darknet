#ifndef TREE_H
#define TREE_H

typedef struct{
    int *leaf;
    int n;
    int *parent;
    char **name;

    int groups;
    int *group_size;
} tree;

tree *read_tree(char *filename);

#endif
