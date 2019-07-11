#ifdef OPENCV
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <cstdio>
#include <time.h>

using namespace cv;
using namespace std;

typedef struct pointNode{
	int x;
	int y;
	struct pointNode* prev;
	struct pointNode* next;
}pointNode;

typedef struct pointList{
	int size;
	pointNode* front;
	pointNode* back;
}pointList;

void load_mat_image_point(char *input);
void initList(pointList* l); // 리스트 초기화
void ListAdd(pointList* l, int x, int y);// 리스트 추가
int ListRemove(pointList* l,int x , int y); // 리스트 제거
void ListPrint(pointList* l);
void ListToArray1(pointList* l, Point* ary);
void ListToArray2(pointList* l, Point **ary);


void onMouse(int event, int x, int y, int flags, void* param);
void onMouseCheck(int event, int x, int y, int flags, void* param);
void draw_line(Mat* im);
void checkIn(Mat* im,int x , int y);
void delay(clock_t sec);
#endif