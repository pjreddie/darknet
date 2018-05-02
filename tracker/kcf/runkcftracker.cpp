
#include "kcftracker.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main(int argc, char **argv)
//int runtracker()
{
    string trackerType = "Native KCF";
	// Create KCFTracker object 
	bool HOG = true,FIXEDWINDOW = false,MULTISCALE = true,LAB = false; //LAB color space features
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);//--------------------------trakcer

    string path = "/media/elab/sdd/data/TLP/Sam";
	// Read the groundtruth bbox
	ifstream groundtruth(path + "/groundtruth_rect.txt");
	int f,x,y,w,h,isLost;
	std::string s;
	getline(groundtruth, s, ',');	
	f = atoi(s.c_str());
	getline(groundtruth, s, ',');
	x = atoi(s.c_str());
	getline(groundtruth, s, ',');	
	y = atoi(s.c_str());
	getline(groundtruth, s, ',');
	w = atoi(s.c_str());
	getline(groundtruth, s, ',');	
	h = atoi(s.c_str());
	getline(groundtruth, s);
	isLost = atoi(s.c_str());
	cout << f <<" " << x <<" " << y <<" " << w <<" " << h <<" " << isLost << endl;
	Rect2d bbox(x,y,w,h);
	Rect2d bboxGroundtruth(x,y,w,h);
	// Read images in a folder
	ostringstream osfile;
	osfile << path << "/img/" << setw(5) << setfill('0') << f <<".jpg";
	cout << osfile.str() << endl;
    cv::Mat frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
    if(! frame.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	// Init the tracker---------------------------tracker
    tracker.init(frame, bbox);

    while(frame.data)
    {   
		// Draw ground truth box
		rectangle(frame, bboxGroundtruth, Scalar( 0, 0, 0 ), 2, 1 );

        // Start timer
        double timer = (double)getTickCount();
         
        // Update the tracking result--------------------------tracker
        bool ok = tracker.update(frame, bbox);
         
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);

        if (ok) {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        } else {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
         
        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
         
        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        // Display frame.        
        //cvNamedWindow("Tracking", CV_WINDOW_NORMAL); 
        //cvShowImage("Tracking", frame);
        imshow("Tracking", frame);
        int c = cvWaitKey(1);
        if (c != -1) c = c%256;
        if (c == 27) {
            cvDestroyWindow("Tracking");
            return 0;
        } 
        waitKey(1);
		// Read next image
		f++;
		osfile.str("");
		osfile << path << "/img/" << setw(5) << setfill('0') << f <<".jpg";
		cout << osfile.str() << endl;
    	frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
		// Read next bbox
		getline(groundtruth, s, ',');	
		f = atoi(s.c_str());
		getline(groundtruth, s, ',');
		x = atoi(s.c_str());
		getline(groundtruth, s, ',');	
		y = atoi(s.c_str());
		getline(groundtruth, s, ',');
		w = atoi(s.c_str());
		getline(groundtruth, s, ',');	
		h = atoi(s.c_str());
		getline(groundtruth, s);
		isLost = atoi(s.c_str());
		cout << f <<" " << x <<" " << y <<" " << w <<" " << h <<" " << isLost << endl;
		bboxGroundtruth.x = x;
		bboxGroundtruth.y = y;
		bboxGroundtruth.width = w;
		bboxGroundtruth.height = h;
    }
    return 0;
}