#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h> /* needed for sockaddr_in */
#include <stdio.h> /* needed for sockaddr_in */
#include <string.h> /* needed for sockaddr_in */
#include <netdb.h>

#include "server.h"
#include "connected_layer.h"

#define MESSAGESIZE 50012
#define NUMFLOATS ((MESSAGESIZE-12)/4)
#define SERVER_PORT 9876
#define CLIENT_PORT 9879
#define STR(x) #x
#define PARAMETER_SERVER localhost

typedef struct{
    int layer;
    int wob;
    int offset;
    float data[NUMFLOATS];
} message;

int socket_setup(int port)
{
    static int fd = 0;                         /* our socket */
    if(fd) return fd;
    struct sockaddr_in myaddr;      /* our address */

    /* create a UDP socket */

    if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("cannot create socket\n");
        fd=0;
        return 0;
    }

    /* bind the socket to any valid IP address and a specific port */

    memset((char *)&myaddr, 0, sizeof(myaddr));
    myaddr.sin_family = AF_INET;
    myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    myaddr.sin_port = htons(port);

    if (bind(fd, (struct sockaddr *)&myaddr, sizeof(myaddr)) < 0) {
        perror("bind failed");
        fd=0;
        return 0;
    }
    return fd;
}

void server_update(network net)
{
    int fd = socket_setup(SERVER_PORT);
    struct sockaddr_in remaddr;     /* remote address */
    socklen_t addrlen = sizeof(remaddr);            /* length of addresses */
    int recvlen;                    /* # bytes received */
    unsigned char buf[MESSAGESIZE];     /* receive buffer */
    message m;

    int count = 0;
    while(1){
        recvlen = recvfrom(fd, buf, MESSAGESIZE, 0, (struct sockaddr *)&remaddr, &addrlen);
        memcpy(&m, buf, recvlen);
        //printf("received %d bytes\n", recvlen);
        //printf("layer %d wob %d offset %d\n", m.layer, m.wob, m.offset);
        ++count;
        if(count % 100 == 0) printf("%d\n", count);
    }
    //printf("%s\n", buf);
}

void client_update(network net)
{
    int fd = socket_setup(CLIENT_PORT);
    struct hostent *hp;     /* host information */
    struct sockaddr_in servaddr;    /* server address */
    printf("%ld %ld\n", sizeof(message), MESSAGESIZE);
    char *my_message = "this is a test message";

    unsigned char buf[MESSAGESIZE];
    message m;

    /* fill in the server's address and data */
    memset((char*)&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(SERVER_PORT);

    /* look up the address of the server given its name */
    hp = gethostbyname("localhost");
    if (!hp) {
        fprintf(stderr, "could not obtain address of %s\n", "localhost");
    }

    /* put the host's address into the server address structure */
    memcpy((void *)&servaddr.sin_addr, hp->h_addr_list[0], hp->h_length);

    /* send a message to the server */
    int i, j, k;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONNECTED){
            connected_layer *layer = (connected_layer *) net.layers[i];
            m.layer = i;
            m.wob = 0;
            for(j = 0; j < layer->outputs; j += NUMFLOATS){
                m.offset = j;

                int num = layer->outputs - j;
                if(NUMFLOATS < num) num = NUMFLOATS;

                memcpy(m.data, &layer->bias_updates[j], num*sizeof(float));
                memcpy(buf, &m, MESSAGESIZE);

                if (sendto(fd, buf, MESSAGESIZE, 0, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
                    perror("sendto failed");
                }
            }
            m.wob = 1;
            for(j = 0; j < layer->outputs*layer->inputs; j += NUMFLOATS){
                m.offset = j;

                int num = layer->outputs*layer->inputs - j;
                if(NUMFLOATS < num) num = NUMFLOATS;

                memcpy(m.data, &layer->weight_updates[j], num*sizeof(float));
                memcpy(buf, &m, MESSAGESIZE);

                if (sendto(fd, buf, MESSAGESIZE, 0, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
                    perror("sendto failed");
                }
            }
        }
    }
}
