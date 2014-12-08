#include <stdio.h> /* needed for sockaddr_in */
#include <string.h> /* needed for sockaddr_in */
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h> /* needed for sockaddr_in */
#include <netdb.h>
#include <pthread.h>

#include "mini_blas.h"
#include "utils.h"
#include "parser.h"
#include "server.h"
#include "connected_layer.h"
#include "convolutional_layer.h"

#define SERVER_PORT 9876
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
    int *counter;
    network net;
} connection_info;

void read_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        int next = read(fd, buffer + n, bytes-n);
        if(next < 0) error("read failed");
        n += next;
    }
}

void write_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        int next = write(fd, buffer + n, bytes-n);
        if(next < 0) error("write failed");
        n += next;
    }
}

void read_and_add_into(int fd, float *a, int n)
{
    float *buff = calloc(n, sizeof(float));
    read_all(fd, (char*) buff, n*sizeof(float));
    axpy_cpu(n, 1, buff, 1, a, 1);
    free(buff);
}

void handle_connection(void *pointer)
{
    printf("New Connection\n");
    connection_info info = *(connection_info *) pointer;
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
    printf("Received updates\n");
    close(fd);
    ++*(info.counter);
    if(*(info.counter)%10==0) save_network(net, "/home/pjreddie/imagenet_backup/alexnet.part");
}

void server_update(network net)
{
    int fd = socket_setup(1);
    int counter = 0;
    listen(fd, 10);
    struct sockaddr_in client;     /* remote address */
    socklen_t client_size = sizeof(client);   /* length of addresses */
    connection_info info;
    info.net = net;
    info.counter = &counter;
    while(1){
        pthread_t worker;
        int connection = accept(fd, (struct sockaddr *) &client, &client_size);
        info.fd = connection;
        pthread_create(&worker, NULL, (void *) &handle_connection, &info);
    }
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

    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *) net.layers[i];

            read_all(fd, (char*) layer.biases, layer.n*sizeof(float));
            int num = layer.n*layer.c*layer.size*layer.size;
            read_all(fd, (char*) layer.filters, num*sizeof(float));

            push_convolutional_layer(layer);
        }
        if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *) net.layers[i];

            read_all(fd, (char *)layer.biases, layer.outputs*sizeof(float));
            read_all(fd, (char *)layer.weights, layer.outputs*layer.inputs*sizeof(float));

            push_connected_layer(layer);
        }
    }
    close(fd);
}
