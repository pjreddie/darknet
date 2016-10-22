#include <stdio.h>
#include <stdlib.h>
#include "tree.h"
#include "utils.h"

tree *read_tree(char *filename)
{
    tree t = {0};
    FILE *fp = fopen(filename, "r");
    
    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while((line=fgetl(fp)) != 0){
        char *id = calloc(256, sizeof(char));
        int parent = -1;
        sscanf(line, "%s %d", id, &parent);
        t.parent = realloc(t.parent, (n+1)*sizeof(int));
        t.parent[n] = parent;
        t.name = realloc(t.name, (n+1)*sizeof(char *));
        t.name[n] = id;
        if(parent != last_parent){
            ++groups;
            t.group_size = realloc(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        ++n;
        ++group_size;
    }
    ++groups;
    t.group_size = realloc(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1] = group_size;
    t.n = n;
    t.groups = groups;
    t.leaf = calloc(n, sizeof(int));
    int i;
    for(i = 0; i < n; ++i) t.leaf[i] = 1;
    for(i = 0; i < n; ++i) if(t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

    fclose(fp);
    tree *tree_ptr = calloc(1, sizeof(tree));
    *tree_ptr = t;
    //error(0);
    return tree_ptr;
}
