#ifdef OPENCV
#include "darknet.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <cstdio>
#include <time.h>

using namespace cv;
using namespace std;

pointList* lists;

void load_mat_image_point(char *input)
{
	Mat image;
	int wait = 10;
	int c;
	lists = (pointList*)calloc(1, sizeof(pointList));
	initList(lists);
    image = imread(input, CV_LOAD_IMAGE_COLOR);
	imshow("Original", image);

	setMouseCallback("Original", onMouse, (void*)(&image));

	c = waitKey(0);
	if (c == 'a')
	{
		setMouseCallback("Original", onMouseCheck, (void*)(&image));
		waitKey(0);
	}
	else if (c == 27)
	{
		return 0;
	}

    ary = 
	return 0;
}
void onMouseCheck(int event, int x, int y, int flags, void* param)
{
	Mat* im = reinterpret_cast<Mat*>(param);
	int i = 0;
	int size;
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		checkIn(im,x,y);
		break;
	}
}
void onMouse(int event, int x, int y, int flags, void* param)
{
	Mat* im = reinterpret_cast<Mat*>(param);
	int i = 0;
	int size;
	switch(event)
	{
		case CV_EVENT_LBUTTONDOWN:
			ListAdd(lists, x, y);
			draw_line(im);
			break;
		case CV_EVENT_RBUTTONDOWN:
			ListRemove(lists, x, y);
			draw_line(im);
			break;
	} 
}

void initList(pointList* l) // 리스트 초기화
{
	l->size = 0;
	l->front = NULL;
	l->back = NULL;
}

void ListAdd(pointList* l, int x, int y)// 리스트 추가
{
	pointNode* news = (pointNode*)calloc(1, sizeof(pointNode));
	news->x = x;
	news->y = y;
	news->next = NULL;
	news->prev = l->back;
	if (l->front == NULL) // empty
	{
		l->front = news;
		l->back = news;
	}
	else
	{
		//printf("news.x = %d , news.y = %d\n", news->x, news->y);
		l->back->next = news;
		l->back = news;
	}
	l->size++;
}
int ListRemove(pointList* l, int x, int y) // 리스트 제거
{
	if (l->front == NULL && l->back == NULL)
	{
		printf("Can't remove\n");
		return -1;
	}
	else if (l->front == l->back)
	{
		l->front = NULL;
		l->back = NULL;
		l->size--;
		return 0;
	}
	else
	{
		pointNode* cur = l->front;
		while (cur != NULL)
		{
			if ((cur->x <= (x + 10) && cur->x >= (x - 10)) && (cur->y <= (y + 10) && cur->y >= (y - 10)))
			{
				if(cur->next !=NULL)
				{
					cur->next->prev = cur->prev;
					cur->prev->next = cur->next;
				}
				if (cur->next == NULL)
				{
					cur->prev->next = NULL;
					l->back = cur->prev;
				} 
				cur->prev = NULL;
				cur->next = NULL;
				l->size--;

			}
			cur = cur->next;
		}
		//free(cur);
		return 1;
	}
}
void ListPrint(pointList* l)
{
	pointNode* cur = l->front;
	if (cur == NULL)
	{
		printf("List is Empty\n");
		return;
	}
	while (cur != NULL)
	{
		printf("X : %d,Y : %d\n", cur->x, cur->y);
		cur = cur->next;
	}
	puts("");
	free(cur);
}

void ListToArray1(pointList* l, Point* ary)
{
	int i = 0;
	pointNode* cur = l->front;
	if (cur == NULL)
	{
		printf("List is Empty\n");
		return;
	}
	while (cur != NULL)
	{
		ary[i++] = Point(cur->x, cur->y);
		cur = cur->next;
	}
	puts("");
	free(cur);
}

void ListToArray2(pointList* l , Point **ary)
{
	int i = 0;
	pointNode* cur = l->front;
	if (cur == NULL)
	{
		printf("List is Empty\n");
		return;
	}
	while (cur != NULL)
	{
		ary[0][i++] = Point(cur->x, cur->y);
		cur = cur->next;
	}
	puts("");
	free(cur);
}

void draw_line(Mat *im)
{
	Point** points;
	int i = 0;
	int size = 0;
	points = (Point * *)calloc(2, sizeof(pointList*));
	for (i = 0; i < 2; i++)
	{
		points[i] = (Point*)calloc(lists->size, sizeof(pointList));
	}
	size = lists->size;
	ListToArray2(lists, points);
	for (i = 0; i < size; i++)
	{
		printf("points->x : %d , points->y : %d\n", points[0][i].x, points[0][i].y);
	}
	const Point* ppt[1] = { points[0] };
	int nsize[1];
	nsize[0] = size;
	*im = imread("./Image/person_259.jpg", CV_LOAD_IMAGE_COLOR);
	polylines(*im, ppt, nsize, 1, true, Scalar(0, 255, 0)); 
	imshow("Original", *im);
	
}


void delay(clock_t sec)
{
	clock_t start = clock();
	while (clock() - start < sec);
}

#endif