#ifdef OPENCV
//
// a single-threaded, multi client(using select), debug webserver - streaming out mjpg.
//  on win, _WIN32 has to be defined, must link against ws2_32.lib (socks on linux are for free)
//

//
// socket related abstractions:
//
#ifdef _WIN32  
#pragma comment(lib, "ws2_32.lib")
#include <winsock.h>
#include <windows.h>
#include <time.h>
#define PORT        unsigned long
#define ADDRPOINTER   int*
struct _INIT_W32DATA
{
	WSADATA w;
	_INIT_W32DATA() { WSAStartup(MAKEWORD(2, 1), &w); }
} _init_once;
#else       /* ! win32 */
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#define PORT        unsigned short
#define SOCKET    int
#define HOSTENT  struct hostent
#define SOCKADDR    struct sockaddr
#define SOCKADDR_IN  struct sockaddr_in
#define ADDRPOINTER  unsigned int*
#define INVALID_SOCKET -1
#define SOCKET_ERROR   -1
#endif /* _WIN32 */

#include <cstdio>
#include <vector>
#include <iostream>
using std::cerr;
using std::endl;

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio.hpp"
#endif
using namespace cv;

#include "http_stream.h"
#include "image.h"


class MJPGWriter
{
	SOCKET sock;
	SOCKET maxfd;
	fd_set master;
	int timeout; // master sock timeout, shutdown after timeout millis.
	int quality; // jpeg compression [1..100]

	int _write(int sock, char const*const s, int len)
	{
		if (len < 1) { len = strlen(s); }
		return ::send(sock, s, len, 0);
	}

public:

	MJPGWriter(int port = 0, int _timeout = 200000, int _quality = 30)
		: sock(INVALID_SOCKET)
		, timeout(_timeout)
		, quality(_quality)
	{
		FD_ZERO(&master);
		if (port)
			open(port);
	}

	~MJPGWriter()
	{
		release();
	}

	bool release()
	{
		if (sock != INVALID_SOCKET)
			::shutdown(sock, 2);
		sock = (INVALID_SOCKET);
		return false;
	}

	bool open(int port)
	{
		sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

		SOCKADDR_IN address;
		address.sin_addr.s_addr = INADDR_ANY;
		address.sin_family = AF_INET;
		address.sin_port = htons(port);	// ::htons(port);
		if (::bind(sock, (SOCKADDR*)&address, sizeof(SOCKADDR_IN)) == SOCKET_ERROR)
		{
			cerr << "error : couldn't bind sock " << sock << " to port " << port << "!" << endl;
			return release();
		}
		if (::listen(sock, 10) == SOCKET_ERROR)
		{
			cerr << "error : couldn't listen on sock " << sock << " on port " << port << " !" << endl;
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
		struct timeval to = { 0,timeout };
		if (::select(maxfd+1, &rread, NULL, NULL, &to) <= 0)
			return true; // nothing broken, there's just noone listening

		std::vector<uchar> outbuf;
		std::vector<int> params;
		params.push_back(IMWRITE_JPEG_QUALITY);
		params.push_back(quality);
		cv::imencode(".jpg", frame, outbuf, params);
		size_t outlen = outbuf.size();

#ifdef _WIN32 
		for (unsigned i = 0; i<rread.fd_count; i++)
		{
			int addrlen = sizeof(SOCKADDR);
			SOCKET s = rread.fd_array[i];    // fd_set on win is an array, while ...
#else         
		for (int s = 0; s<=maxfd; s++)
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
					cerr << "error : couldn't accept connection on sock " << sock << " !" << endl;
					return false;
				}
				maxfd = (maxfd>client ? maxfd : client);
				FD_SET(client, &master);
				_write(client, "HTTP/1.0 200 OK\r\n", 0);
				_write(client,
					"Server: Mozarella/2.2\r\n"
					"Accept-Range: bytes\r\n"
					"Connection: close\r\n"
					"Max-Age: 0\r\n"
					"Expires: 0\r\n"
					"Cache-Control: no-cache, private\r\n"
					"Pragma: no-cache\r\n"
					"Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n"
					"\r\n", 0);
				cerr << "new client " << client << endl;
			}
			else // existing client, just stream pix
			{
				char head[400];
				sprintf(head, "--mjpegstream\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", outlen);
				_write(s, head, 0);
				int n = _write(s, (char*)(&outbuf[0]), outlen);
				//cerr << "known client " << s << " " << n << endl;
				if (n < outlen)
				{
					cerr << "kill client " << s << endl;
					::shutdown(s, 2);
					FD_CLR(s, &master);
				}
			}
		}
		return true;
	}
};
// ----------------------------------------

void send_mjpeg(IplImage* ipl, int port, int timeout, int quality) {
	static MJPGWriter wri(port, timeout, quality);
	cv::Mat mat = cv::cvarrToMat(ipl);
	wri.write(mat);
	std::cout << " MJPEG-stream sent. \n";
}
// ----------------------------------------

CvCapture* get_capture_video_stream(char *path) {
	CvCapture* cap = NULL;
	try {
		cap = (CvCapture*)new cv::VideoCapture(path);
	}
	catch (...) {
		std::cout << " Error: video-stream " << path << " can't be opened! \n";
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
		std::cout << " Error: Web-camera " << index << " can't be opened! \n";
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
			std::cout << " Video-stream stoped! \n";
		}
	}
	catch (...) {
		std::cout << " Video-stream stoped! \n";
	}
	return src;
}
// ----------------------------------------
extern "C" {
	image ipl_to_image(IplImage* src);	// image.c
}

image image_data_augmentation(IplImage* ipl, int w, int h,
	int pleft, int ptop, int swidth, int sheight, int flip,
	float jitter, float dhue, float dsat, float dexp)
{
	cv::Mat img = cv::cvarrToMat(ipl);

	// crop
	cv::Rect src_rect(pleft, ptop, swidth, sheight);
	cv::Rect img_rect(cv::Point2i(0, 0), img.size());
	cv::Rect new_src_rect = src_rect & img_rect;

	cv::Rect dst_rect(cv::Point2i(std::max(0, -pleft), std::max(0, -ptop)), new_src_rect.size());

	cv::Mat cropped(cv::Size(src_rect.width, src_rect.height), img.type());
	cropped.setTo(cv::Scalar::all(0));

	img(new_src_rect).copyTo(cropped(dst_rect));

	// resize
	cv::Mat sized;
	cv::resize(cropped, sized, cv::Size(w, h), 0, 0, INTER_LINEAR);

	// flip
	if (flip) {
		cv::flip(sized, cropped, 1);	// 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
		sized = cropped.clone();
	}

	// HSV augmentation
	// CV_BGR2HSV, CV_RGB2HSV, CV_HSV2BGR, CV_HSV2RGB
	if (ipl->nChannels >= 3)
	{
		cv::Mat hsv_src;
		cvtColor(sized, hsv_src, CV_BGR2HSV);	// also BGR -> RGB
	
		std::vector<cv::Mat> hsv;
		cv::split(hsv_src, hsv);

		hsv[1] *= dsat;
		hsv[2] *= dexp;
		hsv[0] += 179 * dhue;

		cv::merge(hsv, hsv_src);

		cvtColor(hsv_src, sized, CV_HSV2RGB);	// now RGB instead of BGR
	}
	else
	{
		sized *= dexp;
	}

	// Mat -> IplImage -> image
	IplImage src = sized;
	image out = ipl_to_image(&src);

	return out;
}


#endif	// OPENCV
