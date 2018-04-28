# C++ KCF Tracker

To include it in your project, without the VOT toolkit you just need to:
	
	// Create the KCFTracker object with one of the available options
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Give the first frame and the position of the object to the tracker
	tracker.init( Rect(xMin, yMin, width, height), frame );

	// Get the position of the object for the new frame
	result = tracker.update(frame);
