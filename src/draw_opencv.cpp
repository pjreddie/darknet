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
	void onMouseMakeList(int event, int x, int y, int flags, void *param);
	void draw_line(Mat *im);
	void checkIn(Mat *im, int x, int y);
	void delay(clock_t sec);
	void ListToArray1(pointList *l, Point *ary);
	void ListToArray2(pointList *l, Point **ary);
	void checkIn(Mat *im, int x, int y);
	void returnPoint(pointList *l, Points *ary);
	void returnPoints(pointList *l[10], NumPoints *ary);

	pointList *listone;
	pointList *lists[10];
	int c;
	char file_url[512];

	void load_mat_image_points(char *input, int j, NumPoints *ary)
	{
		strcpy(file_url, input);
		puts(file_url);
		Mat image;
		int wait = 10;
		int i = 0;

		for(i = 0 ; i < 10 ; i++)
		{
			lists[i] = (pointList *)calloc(1, sizeof(pointList));
			initList(lists[i]);
		}

		image = imread(input, 1);
		if (image.data)
		{ // can load image
			imshow("Original", image);
			//resizeWindow("Original",image.cols,image.rows);
			printf("image cols : %d, image rows : %d\n", image.cols, image.rows);
			while (1)
			{
				//setMouseCallback("Original", onMouse);

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
				if(c >= '0' && c <= '9')
				{
					setMouseCallback("Original", onMouseMakeList);
					waitKey(0);
					
					if(c <= '9' && c >= '1'){
						Point *points;
						ListToArray1(lists[c-'1'], points);
						for (i = 0; i < lists[c-'1'].size; i++)
						{
							printf("points->x : %d , points->y : %d\n", points[i].x, points[i].y);
							circle(newImage, Point(points[i].x,points[i].y), 5, Scalar(0,0,255), -1);
						}
						returnPoints(lists, ary);
					}
					else if(c == '0')
					{
						returnPoints(lists, ary);
					}
				}
			}
			destroyWindow("Original");
		}
		else
		{
			fprintf(stderr, "Cannot load image \"%s\"\n", file_url);
		}
	}

	void load_mat_image_point(char *input, int i, Points *ary)
	{
		strcpy(file_url, input);
		puts(file_url);
		Mat image;
		int wait = 10;
		listone = (pointList *)calloc(1, sizeof(pointList));
		initList(listone);
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
			returnPoint(listone, ary);
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
			ListAdd(listone, x, y);
			draw_line(im);
			break;
		case CV_EVENT_RBUTTONDOWN:
			ListRemove(listone, x, y);
			draw_line(im);
			break;
		}
	}

	void onMouseMakeList(int event, int x, int y, int flags, void *param)
	{
		Mat *im = reinterpret_cast<Mat *>(param);
		int i = 0;
		int size;
		switch (event)
		{
		case CV_EVENT_LBUTTONDOWN:
			printf("C : %d\n",c-'1');
			if( c >= '1' && c <= '9')
				ListAdd(lists[c-'1'], x, y);
			else
				ListAdd(lists[9],x,y);
			//draw_line(im);
			break;
		case CV_EVENT_RBUTTONDOWN:
			if( c >= '1' && c <= '9')
				ListRemove(lists[c-'1'], x, y);
			else
				ListRemove(lists[9],x,y);
			///draw_line(im);
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
		printf("Check x : %d , y : %d\n",news->x,news->y);
		news->next = NULL;
		news->prev = l->back;
		if (l->front == NULL) // empty
		{
			printf("Add First!!\n");
			l->front = news;
			l->back = news;
		}
		else
		{
			printf("Add Not First!!\n");
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
	/* 수정 필요 */
	void draw_line(Mat *im)
	{
		Point *points;
		int i = 0;
		int size = 0;
		points = (Point *)calloc(listone->size, sizeof(Point));
		size = listone->size;
		ListToArray1(listone, points);
		printf("path : %s\n",file_url);
		Mat newImage = imread(file_url,CV_LOAD_IMAGE_COLOR);
		for (i = 0; i < size; i++)
		{
			printf("points->x : %d , points->y : %d\n", points[i].x, points[i].y);
			circle(newImage, Point(points[i].x,points[i].y), 5, Scalar(0,0,255), -1);
		}
		imshow("Original",newImage);
		free(points);
	}

	void checkIn(Mat *im, int x, int y)
	{
		int crosses = 0;
		Point *points;
		points = (Point *)calloc(listone->size, sizeof(pointList));
		ListToArray1(listone, points);
		int i, j;
		for (i = 0; i < listone->size; i++)
		{
			j = (i + 1) % listone->size;
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

	void returnPoints(pointList *l[10], NumPoints *ary)
	{
		
		ary->size = 0;
		int i = 0;
		int j = 0;
		for( j = 0 ; j < 10 ; j++){
			i = 0;
			pointNode *cur = l[j]->front;
			if (cur == NULL)
			{
				printf("List is Empty\n");
				return;
			}
			while (cur != NULL)
			{
				if (l[j]->size >= 3)
				{
					if(c <= '9' && c >= '1'){
						ary->P[c-'1'].x[i] = cur->x;
						ary->P[c-'1'].y[i++] = cur->y;
						ary->P[c-'1'].size++;
					}
					else if(c == '0')
					{
						ary->P[9].x[i] = cur->x;
						ary->P[9].y[i++] = cur->y;
						ary->P[9].size++;
					}
				}
				else
				{
					printf("Point number is too low\n");
					return;
				}
				cur = cur->next;
			}
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