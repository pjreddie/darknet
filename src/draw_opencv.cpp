#ifdef OPENCV
#include "darknet.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <cstdio>
#include <time.h>
#include <unistd.h>
#include <termio.h>
#include <fcntl.h>

using namespace cv;
using namespace std;

extern "C"
{
	void onMouse(int event, int x, int y, int flags, void *param);
	void onMouseCheck(int event, int x, int y, int flags, void *param);
	void draw_line(Mat *im);
	void checkIn(Mat *im, int x, int y);
	void delay(clock_t sec);
	void ListToArray1(pointList *l, Point *ary);
	void ListToArray2(pointList *l, Point **ary);
	void checkIn(Mat *im, int x, int y);
	void returnPoint(pointList *l, Points *ary);

	pointList *lists;

	char file_url[512];

	void load_mat_image_point(char *input, int i, Points *ary)
	{
		strcpy(file_url, input);
		puts(file_url);
		Mat image;
		int wait = 10;
		int c;
		lists = (pointList *)calloc(1, sizeof(pointList));
		initList(lists);
		image = imread(input, 1);
		if (image.data)
		{ // can load image
			imshow("Original", image);
			//resizeWindow("Original",image.cols,image.rows);
			printf("image cols : %d, image rows : %d\n", image.cols, image.rows);
			while (1)
			{
				setMouseCallback("Original", onMouse);

				c = waitKey(0);
				if (c == 'a')
				{
					setMouseCallback("Original", onMouseCheck);
					waitKey(0);
				}
				if (c == 32)
				{
					break;
				}
			}
			destroyWindow("Original");
			returnPoint(lists, ary);
		}
		else
		{
			fprintf(stderr, "Cannot load image \"%s\"\n", file_url);
		}
	}
	void onMouseCheck(int event, int x, int y, int flags, void *param)
	{
		Mat *im = reinterpret_cast<Mat *>(param);
		int i = 0;
		int size;
		switch (event)
		{
		case CV_EVENT_LBUTTONDOWN:
			checkIn(im, x, y);
			break;
		}
	}
	void onMouse(int event, int x, int y, int flags, void *param)
	{
		Mat *im = reinterpret_cast<Mat *>(param);
		int i = 0;
		int size;
		switch (event)
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

	void initList(pointList *l) // 리스트 초기화
	{
		l->size = 0;
		l->front = NULL;
		l->back = NULL;
	}

	void ListAdd(pointList *l, int x, int y) // 리스트 추가
	{
		pointNode *news = (pointNode *)calloc(1, sizeof(pointNode));
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
	int ListRemove(pointList *l, int x, int y) // 리스트 제거
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
			pointNode *cur = l->front;
			while (cur != NULL)
			{
				if ((cur->x <= (x + 10) && cur->x >= (x - 10)) && (cur->y <= (y + 10) && cur->y >= (y - 10)))
				{
					if (cur->next != NULL)
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
	void ListPrint(pointList *l)
	{
		pointNode *cur = l->front;
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

	void ListToArray1(pointList *l, Point *ary)
	{
		int i = 0;
		pointNode *cur = l->front;
		if (cur == NULL)
		{
			printf("List is Empty\n");
			return;
		}
		while (cur != NULL)
		{
			ary[i].x = cur->x;
			ary[i++].y = cur->y;
			cur = cur->next;
		}
		puts("");
		free(cur);
	}

	void ListToArray2(pointList *l, Point **ary)
	{
		int i = 0;
		pointNode *cur = l->front;
		if (cur == NULL)
		{
			printf("List is Empty\n");
			return;
		}
		while (cur != NULL)
		{
			ary[0][i].x = cur->x;
			ary[0][i++].y = cur->y;
			cur = cur->next;
		}
		puts("");
		free(cur);
	}

	void draw_line(Mat *im)
	{
		Point *points;
		int i = 0;
		int size = 0;
		points = (Point *)calloc(lists->size, sizeof(Point));
		size = lists->size;
		ListToArray1(lists, points);
		for (i = 0; i < size; i++)
		{
			printf("points->x : %d , points->y : %d\n", points[i].x, points[i].y);
			// 해당 부분 해결 방법 강구하기
			//puts(file_url);
			/*
			int j, z;
			for (j = points[i].x - 1; j <= points[i].x + 1; j++)
			{
				for (z = points[j].y - 1; z <= points[i].y + 1; z++)
				{
					im->at<Vec3b>(j, z)[0] = 0;   // Blue
					im->at<Vec3b>(j, z)[1] = 0;   // Green
					im->at<Vec3b>(j, z)[2] = 255; // Red
				}
			}
 			*/
			free(points);
		}
	}
	void delay(clock_t sec)
	{
		clock_t start = clock();
		while (clock() - start < sec)
			;
	}

	void checkIn(Mat *im, int x, int y)
	{
		int crosses = 0;
		Point *points;
		points = (Point *)calloc(lists->size, sizeof(pointList));
		ListToArray1(lists, points);
		int i, j;
		for (i = 0; i < lists->size; i++)
		{
			j = (i + 1) % lists->size;
			if ((points[i].y > y) != (points[j].y > y)) // 두 좌표(연결점)의 y좌표가 점의 좌표와 교차할 경우만 확인
			{
				double atX = (points[j].x - points[i].x) * (y - points[i].y) / (points[j].y - points[i].y) + points[i].x;
				if (x <= atX)
					crosses++;
			}
		}
		if (crosses % 2 == 1)
		{
			printf("내부의점\n");
		}
		else
		{
			printf("외부의점\n");
		}
	}

	void returnPoint(pointList *l, Points *ary)
	{
		ary->size = 0;
		int i = 0;
		pointNode *cur = l->front;
		if (cur == NULL)
		{
			printf("List is Empty\n");
			return;
		}
		while (cur != NULL)
		{
			if (l->size >= 3)
			{
				ary->x[i] = cur->x;
				ary->y[i++] = cur->y;
				//printf("ary->x[%d] : %d , ary->y[%d] : %d\n",i,ary->x[i-1],i,ary->y[i-1]);
				//printf("ary->size : %d\n",ary->size);
				ary->size++;
			}
			else
			{
				printf("Point number is too low\n");
				return;
			}
			cur = cur->next;
		}
		puts("");
		//free(cur);
	}
	int check_person_point(int px, int py, Points *ary)
	{
		int crosses = 0;
		int i, j;
		for (i = 0; i < ary->size; i++)
		{
			j = (i + 1) % ary->size;
			if ((ary->y[i] > py) != (ary->y[j] > py)) // 두 좌표(연결점)의 y좌표가 점의 좌표와 교차할 경우만 확인
			{
				double atX = (ary->x[j] - ary->x[i]) * (py - ary->y[i]) / (ary->y[j] - ary->y[i]) + ary->x[i];
				if (px <= atX)
					crosses++;
			}
		}
		if (crosses % 2 == 1)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
}
#endif