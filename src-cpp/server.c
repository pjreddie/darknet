#include <stdio.h> /* needed for sockaddr_in */
#include <string.h> /* needed for sockaddr_in */
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h> /* needed for sockaddr_in */
#include <netdb.h>
#include <pthread.h>
#include <time.h>

#include "mini_blas.h"
#include "utils.h"
#include "parser.h"
#include "server.h"
#include "connected_layer.h"
#include "convolutional_layer.h"

#define SERVER_PORT 9423
#define STR(x) #x

int socket_setup(int server)
{
    int fd = 0;                         /* our socket */
    struct sockaddr_in me;      /* our address */

    /* create a UDP socket */

    if ((fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        error("cannot create socket");
    }

    /* bind the socket to any valid IP address and a specific port */
    if (server == 1){
        bzero((char *) &me, sizeof(me));
        me.sin_family = AF_INET;
        me.sin_addr.s_addr = htonl(INADDR_ANY);
        me.sin_port = htons(SERVER_PORT);

        if (bind(fd, (struct sockaddr *)&me, sizeof(me)) < 0) {
            error("bind failed");
        }
    }

    return fd;
}

typedef struct{
    int fd;
    int counter;
    network net;
} connection_info;

void read_and_add_into(int fd, float *a, int n)
{
    float *buff = calloc(n, sizeof(float));
    read_all(fd, (char*) buff, n*sizeof(float));
    axpy_cpu(n, 1, buff, 1, a, 1);
    free(buff);
}

void handle_connection(void *pointer)
{
    connection_info info = *(connection_info *) pointer;
    free(pointer);
    //printf("New Connection\n");
    if(info.counter%100==0){
        char buff[256];
        sprintf(buff, "unikitty/net_%d.part", info.counter);
        save_network(info.net, buff);
    }
    int fd = info.fd;
    network net = info.net;
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *) net.layers[i];

            read_and_add_into(fd, layer.bias_updates, layer.n);
            int num = layer.n*layer.c*layer.size*layer.size;
            read_and_add_into(fd, layer.filter_updates, num);
        }
        if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *) net.layers[i];

            read_and_add_into(fd, layer.bias_updates, layer.outputs);
            read_and_add_into(fd, layer.weight_updates, layer.inputs*layer.outputs);
        }
    }
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *) net.layers[i];
            update_convolutional_layer(layer);

            write_all(fd, (char*) layer.biases, layer.n*sizeof(float));
            int num = layer.n*layer.c*layer.size*layer.size;
            write_all(fd, (char*) layer.filters, num*sizeof(float));
        }
        if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *) net.layers[i];
            update_connected_layer(layer);
            write_all(fd, (char *)layer.biases, layer.outputs*sizeof(float));
            write_all(fd, (char *)layer.weights, layer.outputs*layer.inputs*sizeof(float));
        }
    }
    //printf("Received updates\n");
    close(fd);
}

void server_update(network net)
{
    int fd = socket_setup(1);
    int counter = 18000;
    listen(fd, 64);
    struct sockaddr_in client;     /* remote address */
    socklen_t client_size = sizeof(client);   /* length of addresses */
    time_t t=0;
    while(1){
        connection_info *info = calloc(1, sizeof(connection_info));
        info->net = net;
        info->counter = counter;
        pthread_t worker;
        int connection = accept(fd, (struct sockaddr *) &client, &client_size);
        if(!t) t=time(0);
        info->fd = connection;
        pthread_create(&worker, NULL, (void *) &handle_connection, info);
        ++counter;
        printf("%d\n", counter);
        //if(counter == 1024) break;
    }
    close(fd);
}

void client_update(network net, char *address)
{
    int fd = socket_setup(0);

    struct hostent *hp;     /* host information */
    struct sockaddr_in server;    /* server address */

    /* fill in the server's address and data */
    bzero((char*)&server, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(SERVER_PORT);

    /* look up the address of the server given its name */
    hp = gethostbyname(address);
    if (!hp) {
        perror("no such host");
        fprintf(stderr, "could not obtain address of %s\n", "localhost");
    }

    /* put the host's address into the server address structure */
    memcpy((void *)&server.sin_addr, hp->h_addr_list[0], hp->h_length);
    if (connect(fd, (struct sockaddr *) &server, sizeof(server)) < 0) {
        error("error connecting");
    }

    /* send a message to the server */
    int i;
    //printf("Sending\n");
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *) net.layers[i];
            write_all(fd, (char*) layer.bias_updates, layer.n*sizeof(float));
            int num = layer.n*layer.c*layer.size*layer.size;
            write_all(fd, (char*) layer.filter_updates, num*sizeof(float));
            memset(layer.bias_updates, 0, layer.n*sizeof(float));
            memset(layer.filter_updates, 0, num*sizeof(float));
        }
        if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *) net.layers[i];
            write_all(fd, (char *)layer.bias_updates, layer.outputs*sizeof(float));
            write_all(fd, (char *)layer.weight_updates, layer.outputs*layer.inputs*sizeof(float));
            memset(layer.bias_updates, 0, layer.outputs*sizeof(float));
            memset(layer.weight_updates, 0, layer.inputs*layer.outputs*sizeof(float));
        }
    }
    //printf("Sent\n");

    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *) net.layers[i];

            read_all(fd, (char*) layer.biases, layer.n*sizeof(float));
            int num = layer.n*layer.c*layer.size*layer.size;
            read_all(fd, (char*) layer.filters, num*sizeof(float));

#ifdef GPU
            push_convolutional_layer(layer);
            #endif
        }
        if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *) net.layers[i];

            read_all(fd, (char *)layer.biases, layer.outputs*sizeof(float));
            read_all(fd, (char *)layer.weights, layer.outputs*layer.inputs*sizeof(float));

#ifdef GPU
            push_connected_layer(layer);
            #endif
        }
    }
    //printf("Updated\n");
    close(fd);
}
