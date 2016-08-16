#ifndef SORTTRACKING
#define SORTTRACKING

/**This file implements the main tracking algorithm ,sort tracking
   Please find the paper: Simple online and realtime tracking
   ------------MORE THAN THE PAPER-----------
**/
#pragma once
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"


#include "itf/trackers/utils.h"
#include "KalmanBoxTracker.hpp"
#include "HungarianAlign.hpp"




class Track
{
public:
	std::vector<data> trace;
        bool isCrossLine = false;
        bool isCounted = false;
	static std::size_t NextTrackID;
	std::size_t track_id;
	std::size_t skipped_frames; 
	data prediction;
	KalmanBoxTracker* KF;
	Track(data p, float dt, float Accel_noise_mag, bool reID, int ID);
	~Track();
};


class Sort
{
public:
	
	float dt; 

	float Accel_noise_mag;
	int maximum_allowed_skipped_frames;
	double max_time;
	int max_trace_length;

	vector<Track*> tracks;
	std::vector<data> delTracks;
	void Update(vector<data>& detections,cv:: Mat frame,unsigned long frameNumber);
        unsigned int imageW;
        unsigned int imageH;
	Sort(float _dt, float _Accel_noise_mag,int _maximum_allowed_skipped_frames=5,double _max_time=0.05,int _max_trace_length=10,unsigned int w=720,unsigned int h=560); //default value
	~Sort(void);
};

#endif
