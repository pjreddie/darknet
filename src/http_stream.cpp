#include "image.h"
#include "http_stream.h"

#ifdef OPENCV
//
// a single-threaded, multi client(using select), debug webserver - streaming out mjpg.
//  on win, _WIN32 has to be defined, must link against ws2_32.lib (socks on linux are for free)
//

#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using std::cerr;
using std::endl;

//
// socket related abstractions:
//
#ifdef _WIN32
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "ws2_32.lib")
#endif
#include "gettimeofday.h"
#include <time.h>
#define PORT        unsigned long
#define ADDRPOINTER   int*
struct _INIT_W32DATA
{
    WSADATA w;
    _INIT_W32DATA() { WSAStartup(MAKEWORD(2, 1), &w); }
} _init_once;

// Graceful closes will first close their output channels and then wait for the peer
// on the other side of the connection to close its output channels. When both sides are done telling
// each other they won,t be sending any more data (i.e., closing output channels),
// the connection can be closed fully, with no risk of reset.
static int close_socket(SOCKET s) {
    int close_output = ::shutdown(s, 1); // 0 close input, 1 close output, 2 close both
    char *buf = (char *)calloc(1024, sizeof(char));
    ::recv(s, buf, 1024, 0);
    free(buf);
    int close_input = ::shutdown(s, 0);
    int result = ::closesocket(s);
    cerr << "Close socket: out = " << close_output << ", in = " << close_input << " \n";
    return result;
}
#else   // nix
#include "darkunistd.h"
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#define PORT        unsigned short
#define SOCKET    int
#define HOSTENT  struct hostent
#define SOCKADDR    struct sockaddr
#define SOCKADDR_IN  struct sockaddr_in
#define ADDRPOINTER  unsigned int*
#define INVALID_SOCKET -1
#define SOCKET_ERROR   -1
struct _IGNORE_PIPE_SIGNAL
{
    struct sigaction new_actn, old_actn;
    _IGNORE_PIPE_SIGNAL() {
        new_actn.sa_handler = SIG_IGN;  // ignore the broken pipe signal
        sigemptyset(&new_actn.sa_mask);
        new_actn.sa_flags = 0;
        sigaction(SIGPIPE, &new_actn, &old_actn);
        // sigaction (SIGPIPE, &old_actn, NULL); // - to restore the previous signal handling
    }
} _init_once;

static int close_socket(SOCKET s) {
    int close_output = ::shutdown(s, 1); // 0 close input, 1 close output, 2 close both
    char *buf = (char *)calloc(1024, sizeof(char));
    ::recv(s, buf, 1024, 0);
    free(buf);
    int close_input = ::shutdown(s, 0);
    int result = close(s);
    std::cerr << "Close socket: out = " << close_output << ", in = " << close_input << " \n";
    return result;
}
#endif // _WIN32


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#ifndef CV_VERSION_EPOCH
#include <opencv2/videoio/videoio.hpp>
#endif
using namespace cv;



class MJPG_sender
{
    SOCKET sock;
    SOCKET maxfd;
    fd_set master;
    int timeout; // master sock timeout, shutdown after timeout usec.
    int quality; // jpeg compression [1..100]
    int close_all_sockets;

    int _write(int sock, char const*const s, int len)
    {
        if (len < 1) { len = strlen(s); }
        return ::send(sock, s, len, 0);
    }

public:

    MJPG_sender(int port = 0, int _timeout = 200000, int _quality = 30)
        : sock(INVALID_SOCKET)
        , timeout(_timeout)
        , quality(_quality)
    {
        close_all_sockets = 0;
        FD_ZERO(&master);
        if (port)
            open(port);
    }

    ~MJPG_sender()
    {
        close_all();
        release();
    }

    bool release()
    {
        if (sock != INVALID_SOCKET)
            ::shutdown(sock, 2);
        sock = (INVALID_SOCKET);
        return false;
    }

    void close_all()
    {
        close_all_sockets = 1;
        cv::Mat tmp(cv::Size(10, 10), CV_8UC3);
        write(tmp);
    }

    bool open(int port)
    {
        sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        SOCKADDR_IN address;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_family = AF_INET;
        address.sin_port = htons(port);    // ::htons(port);
        int reuse = 1;
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse)) < 0)
            cerr << "setsockopt(SO_REUSEADDR) failed" << endl;

#ifdef SO_REUSEPORT
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse)) < 0)
            cerr << "setsockopt(SO_REUSEPORT) failed" << endl;
#endif
        if (::bind(sock, (SOCKADDR*)&address, sizeof(SOCKADDR_IN)) == SOCKET_ERROR)
        {
            cerr << "error MJPG_sender: couldn't bind sock " << sock << " to port " << port << "!" << endl;
            return release();
        }
        if (::listen(sock, 10) == SOCKET_ERROR)
        {
            cerr << "error MJPG_sender: couldn't listen on sock " << sock << " on port " << port << " !" << endl;
            return release();
        }
        FD_ZERO(&master);
        FD_SET(sock, &master);
        maxfd = sock;
        return true;
    }

    bool isOpened()
    {
        return sock != INVALID_SOCKET;
    }

    bool write(const Mat & frame)
    {
        fd_set rread = master;
        struct timeval select_timeout = { 0, 0 };
        struct timeval socket_timeout = { 0, timeout };
        if (::select(maxfd + 1, &rread, NULL, NULL, &select_timeout) <= 0)
            return true; // nothing broken, there's just noone listening

        std::vector<uchar> outbuf;
        std::vector<int> params;
        params.push_back(IMWRITE_JPEG_QUALITY);
        params.push_back(quality);
        //cv::imencode(".jpg", frame, outbuf, params);  //REMOVED FOR COMPATIBILITY
        std::cerr << "cv::imencode call disabled!" << std::endl;
        size_t outlen = outbuf.size();

#ifdef _WIN32
        for (unsigned i = 0; i<rread.fd_count; i++)
        {
            int addrlen = sizeof(SOCKADDR);
            SOCKET s = rread.fd_array[i];    // fd_set on win is an array, while ...
#else
        for (int s = 0; s <= maxfd; s++)
        {
            socklen_t addrlen = sizeof(SOCKADDR);
            if (!FD_ISSET(s, &rread))      // ... on linux it's a bitmask ;)
                continue;
#endif
            if (s == sock) // request on master socket, accept and send main header.
            {
                SOCKADDR_IN address = { 0 };
                SOCKET      client = ::accept(sock, (SOCKADDR*)&address, &addrlen);
                if (client == SOCKET_ERROR)
                {
                    cerr << "error MJPG_sender: couldn't accept connection on sock " << sock << " !" << endl;
                    return false;
                }
                if (setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (char *)&socket_timeout, sizeof(socket_timeout)) < 0) {
                    cerr << "error MJPG_sender: SO_RCVTIMEO setsockopt failed\n";
                }
                if (setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, (char *)&socket_timeout, sizeof(socket_timeout)) < 0) {
                    cerr << "error MJPG_sender: SO_SNDTIMEO setsockopt failed\n";
                }
                maxfd = (maxfd>client ? maxfd : client);
                FD_SET(client, &master);
                _write(client, "HTTP/1.0 200 OK\n", 0);
                _write(client,
                    "Server: Mozarella/2.2\n"
                    "Accept-Range: bytes\n"
                    "Connection: close\n"
                    "Max-Age: 0\n"
                    "Expires: 0\n"
                    "Cache-Control: no-cache, private\n"
                    "Pragma: no-cache\n"
                    "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\n"
                    "\n", 0);
                cerr << "MJPG_sender: new client " << client << endl;
            }
            else // existing client, just stream pix
            {
                if (close_all_sockets) {
                    int result = close_socket(s);
                    cerr << "MJPG_sender: close clinet: " << result << " \n";
                    continue;
                }

                char head[400];
                sprintf(head, "--mjpegstream\nContent-Type: image/jpeg\nContent-Length: %zu\n\n", outlen);
                _write(s, head, 0);
                int n = _write(s, (char*)(&outbuf[0]), outlen);
                //cerr << "known client " << s << " " << n << endl;
                if (n < outlen)
                {
                    cerr << "MJPG_sender: kill client " << s << endl;
                    ::shutdown(s, 2);
                    FD_CLR(s, &master);
                }
            }
        }
        if (close_all_sockets) {
            int result = close_socket(sock);
            cerr << "MJPG_sender: close acceptor: " << result << " \n\n";
        }
        return true;
    }
};
// ----------------------------------------

void send_mjpeg(IplImage* ipl, int port, int timeout, int quality)
{
    try {
        static MJPG_sender wri(port, timeout, quality);
        cv::Mat mat = cv::cvarrToMat(ipl);
        wri.write(mat);
        std::cout << " MJPEG-stream sent. \n";
    }
    catch (...) {
        cerr << " Error in send_mjpeg() function \n";
    }
}
// ----------------------------------------

class JSON_sender
{
    SOCKET sock;
    SOCKET maxfd;
    fd_set master;
    int timeout; // master sock timeout, shutdown after timeout usec.
    int close_all_sockets;

    int _write(int sock, char const*const s, int len)
    {
        if (len < 1) { len = strlen(s); }
        return ::send(sock, s, len, 0);
    }

public:

    JSON_sender(int port = 0, int _timeout = 200000)
        : sock(INVALID_SOCKET)
        , timeout(_timeout)
    {
        close_all_sockets = 0;
        FD_ZERO(&master);
        if (port)
            open(port);
    }

    ~JSON_sender()
    {
        close_all();
        release();
    }

    bool release()
    {
        if (sock != INVALID_SOCKET)
            ::shutdown(sock, 2);
        sock = (INVALID_SOCKET);
        return false;
    }

    void close_all()
    {
        close_all_sockets = 1;
        write("\n]");   // close JSON array
    }

    bool open(int port)
    {
        sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        SOCKADDR_IN address;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_family = AF_INET;
        address.sin_port = htons(port);    // ::htons(port);
        int reuse = 1;
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse)) < 0)
            cerr << "setsockopt(SO_REUSEADDR) failed" << endl;

#ifdef SO_REUSEPORT
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse)) < 0)
            cerr << "setsockopt(SO_REUSEPORT) failed" << endl;
#endif
        if (::bind(sock, (SOCKADDR*)&address, sizeof(SOCKADDR_IN)) == SOCKET_ERROR)
        {
            cerr << "error JSON_sender: couldn't bind sock " << sock << " to port " << port << "!" << endl;
            return release();
        }
        if (::listen(sock, 10) == SOCKET_ERROR)
        {
            cerr << "error JSON_sender: couldn't listen on sock " << sock << " on port " << port << " !" << endl;
            return release();
        }
        FD_ZERO(&master);
        FD_SET(sock, &master);
        maxfd = sock;
        return true;
    }

    bool isOpened()
    {
        return sock != INVALID_SOCKET;
    }

    bool write(char *outputbuf)
    {
        fd_set rread = master;
        struct timeval select_timeout = { 0, 0 };
        struct timeval socket_timeout = { 0, timeout };
        if (::select(maxfd + 1, &rread, NULL, NULL, &select_timeout) <= 0)
            return true; // nothing broken, there's just noone listening

        size_t outlen = strlen(outputbuf);

#ifdef _WIN32
        for (unsigned i = 0; i<rread.fd_count; i++)
        {
            int addrlen = sizeof(SOCKADDR);
            SOCKET s = rread.fd_array[i];    // fd_set on win is an array, while ...
#else
        for (int s = 0; s <= maxfd; s++)
        {
            socklen_t addrlen = sizeof(SOCKADDR);
            if (!FD_ISSET(s, &rread))      // ... on linux it's a bitmask ;)
                continue;
#endif
            if (s == sock) // request on master socket, accept and send main header.
            {
                SOCKADDR_IN address = { 0 };
                SOCKET      client = ::accept(sock, (SOCKADDR*)&address, &addrlen);
                if (client == SOCKET_ERROR)
                {
                    cerr << "error JSON_sender: couldn't accept connection on sock " << sock << " !" << endl;
                    return false;
                }
                if (setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (char *)&socket_timeout, sizeof(socket_timeout)) < 0) {
                    cerr << "error JSON_sender: SO_RCVTIMEO setsockopt failed\n";
                }
                if (setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, (char *)&socket_timeout, sizeof(socket_timeout)) < 0) {
                    cerr << "error JSON_sender: SO_SNDTIMEO setsockopt failed\n";
                }
                maxfd = (maxfd>client ? maxfd : client);
                FD_SET(client, &master);
                _write(client, "HTTP/1.0 200 OK\n", 0);
                _write(client,
                    "Server: Mozarella/2.2\n"
                    "Accept-Range: bytes\n"
                    "Connection: close\n"
                    "Max-Age: 0\n"
                    "Expires: 0\n"
                    "Cache-Control: no-cache, private\n"
                    "Pragma: no-cache\n"
                    "Content-Type: application/json\n"
                    //"Content-Type: multipart/x-mixed-replace; boundary=boundary\r\n"
                    "\n", 0);
                _write(client, "[\n", 0);   // open JSON array
                int n = _write(client, outputbuf, outlen);
                cerr << "JSON_sender: new client " << client << endl;
            }
            else // existing client, just stream pix
            {
                //char head[400];
                // application/x-resource+json or application/x-collection+json -  when you are representing REST resources and collections
                // application/json or text/json or text/javascript or text/plain.
                // https://stackoverflow.com/questions/477816/what-is-the-correct-json-content-type
                //sprintf(head, "\r\nContent-Length: %zu\r\n\r\n", outlen);
                //sprintf(head, "--boundary\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n", outlen);
                //_write(s, head, 0);
                if(!close_all_sockets) _write(s, ", \n", 0);
                int n = _write(s, outputbuf, outlen);
                if (n < outlen)
                {
                    cerr << "JSON_sender: kill client " << s << endl;
                    ::shutdown(s, 2);
                    FD_CLR(s, &master);
                }

                if (close_all_sockets) {
                    int result = close_socket(s);
                    cerr << "JSON_sender: close clinet: " << result << " \n";
                    continue;
                }
            }
        }
        if (close_all_sockets) {
            int result = close_socket(sock);
            cerr << "JSON_sender: close acceptor: " << result << " \n\n";
        }
        return true;
    }
};
// ----------------------------------------

void send_json(detection *dets, int nboxes, int classes, char **names, long long int frame_id, int port, int timeout)
{
    try {
        static JSON_sender js(port, timeout);
        char *send_buf = detection_to_json(dets, nboxes, classes, names, frame_id, NULL);

        js.write(send_buf);
        std::cout << " JSON-stream sent. \n";
        free(send_buf);
    }
    catch (...) {
        cerr << " Error in send_json() function \n";
    }
}

// ----------------------------------------

CvCapture* get_capture_video_stream(const char *path) {
    CvCapture* cap = NULL;
    try {
        cap = (CvCapture*)new cv::VideoCapture(path);
    }
    catch (...) {
        cerr << " Error: video-stream " << path << " can't be opened! \n";
    }
    return cap;
}
// ----------------------------------------

CvCapture* get_capture_webcam(int index) {
    CvCapture* cap = NULL;
    try {
        cap = (CvCapture*)new cv::VideoCapture(index);
        //((cv::VideoCapture*)cap)->set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        //((cv::VideoCapture*)cap)->set(CV_CAP_PROP_FRAME_HEIGHT, 960);
    }
    catch (...) {
        cerr << " Error: Web-camera " << index << " can't be opened! \n";
    }
    return cap;
}
// ----------------------------------------

IplImage* get_webcam_frame(CvCapture *cap) {
    IplImage* src = NULL;
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
        cv::Mat frame;
        if (cpp_cap.isOpened())
        {
            cpp_cap >> frame;
            IplImage tmp = frame;
            src = cvCloneImage(&tmp);
        }
        else {
            std::cout << " Video-stream stopped! \n";
        }
    }
    catch (...) {
        std::cout << " Video-stream stoped! \n";
    }
    return src;
}

int get_stream_fps_cpp(CvCapture *cap) {
    int fps = 25;
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        fps = cpp_cap.get(CAP_PROP_FPS);
#else                        // OpenCV 2.x
        fps = cpp_cap.get(CV_CAP_PROP_FPS);
#endif
    }
    catch (...) {
        cerr << " Can't get FPS of source videofile. For output video FPS = 25 by default. \n";
    }
    return fps;
}
// ----------------------------------------

image image_data_augmentation(IplImage* ipl, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float jitter, float dhue, float dsat, float dexp)
{
    image out;
    try {
        cv::Mat img = cv::cvarrToMat(ipl);

        // crop
        cv::Rect src_rect(pleft, ptop, swidth, sheight);
        cv::Rect img_rect(cv::Point2i(0, 0), img.size());
        cv::Rect new_src_rect = src_rect & img_rect;

        cv::Rect dst_rect(cv::Point2i(std::max<int>(0, -pleft), std::max<int>(0, -ptop)), new_src_rect.size());

        cv::Mat cropped(cv::Size(src_rect.width, src_rect.height), img.type());
        cropped.setTo(cv::Scalar::all(0));

        img(new_src_rect).copyTo(cropped(dst_rect));

        // resize
        cv::Mat sized;
        cv::resize(cropped, sized, cv::Size(w, h), 0, 0, INTER_LINEAR);

        // flip
        if (flip) {
            cv::flip(sized, cropped, 1);    // 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
            sized = cropped.clone();
        }

        // HSV augmentation
        // CV_BGR2HSV, CV_RGB2HSV, CV_HSV2BGR, CV_HSV2RGB
        if (ipl->nChannels >= 3)
        {
            cv::Mat hsv_src;
            cvtColor(sized, hsv_src, CV_BGR2HSV);    // also BGR -> RGB

            std::vector<cv::Mat> hsv;
            cv::split(hsv_src, hsv);

            hsv[1] *= dsat;
            hsv[2] *= dexp;
            hsv[0] += 179 * dhue;

            cv::merge(hsv, hsv_src);

            cvtColor(hsv_src, sized, CV_HSV2RGB);    // now RGB instead of BGR
        }
        else
        {
            sized *= dexp;
        }

        //std::stringstream window_name;
        //window_name << "augmentation - " << ipl;
        //cv::imshow(window_name.str(), sized);
        //cv::waitKey(0);

        // Mat -> IplImage -> image
        IplImage src = sized;
        out = ipl_to_image(&src);
    }
    catch (...) {
        cerr << "OpenCV can't augment image: " << w  << " x " << h << " \n";
        out = ipl_to_image(ipl);
    }
    return out;
}


image load_image_resize(char *filename, int w, int h, int c, image *im)
{
    image out;
    cv::Mat img(h, w, CV_8UC3);
    try {
        int flag = -1;
        if (c == 0) flag = 1;
        else if (c == 1) { flag = 0; img = cv::Mat(h, w, CV_8UC1); }
        else if (c == 3) { flag = 1; img = cv::Mat(h, w, CV_8UC3); }
        else {
            cerr << "OpenCV can't force load with " << c << " channels\n";
        }
        //throw std::runtime_error("runtime_error");
        cv::Mat loaded_image = cv::imread(filename, flag);
        cv::cvtColor(loaded_image, loaded_image, cv::COLOR_RGB2BGR);
        IplImage tmp1 = loaded_image;
        *im = ipl_to_image(&tmp1);

        cv::resize(loaded_image, img, cv::Size(w, h), 0, 0, CV_INTER_LINEAR);

        IplImage tmp2 = img;
        out = ipl_to_image(&tmp2);
    }
    catch (...) {
        cerr << "OpenCV can't load image %s " << filename << " \n";
        out = make_image(w, h, c);
        *im = make_image(w, h, c);
    }
    return out;
}


#endif    // OPENCV

// -----------------------------------------------------

#if __cplusplus >= 201103L || _MSC_VER >= 1900  // C++11

#include <chrono>
#include <iostream>

static std::chrono::steady_clock::time_point steady_start, steady_end;
static double total_time;

double get_time_point() {
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    //uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(current_time.time_since_epoch()).count();
    return std::chrono::duration_cast<std::chrono::microseconds>(current_time.time_since_epoch()).count();
}

void start_timer() {
    steady_start = std::chrono::steady_clock::now();
}

void stop_timer() {
    steady_end = std::chrono::steady_clock::now();
}

double get_time() {
    double took_time = std::chrono::duration<double>(steady_end - steady_start).count();
    total_time += took_time;
    return took_time;
}

void stop_timer_and_show() {
    stop_timer();
    std::cout << " " << get_time() * 1000 << " msec" << std::endl;
}

void stop_timer_and_show_name(char *name) {
    stop_timer();
    std::cout << " " << name;
    std::cout << " " << get_time() * 1000 << " msec" << std::endl;
}

void show_total_time() {
    std::cout << " Total: " << total_time * 1000 << " msec" << std::endl;
}

#else // C++11
#include <iostream>

double get_time_point() { return 0; }
void start_timer() {}
void stop_timer() {}
double get_time() { return 0; }
void stop_timer_and_show() {
    std::cout << " stop_timer_and_show() isn't implemented " << std::endl;
}
void stop_timer_and_show_name(char *name) { stop_timer_and_show(); }
void total_time() {}
#endif // C++11
