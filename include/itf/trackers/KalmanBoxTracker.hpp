#ifndef KALMANBOXTRACKER
#define KALMANBOXTRACKER

#include "itf/trackers/utils.h"

#include "opencv2/opencv.hpp"
#include <opencv/cv.h>


class KalmanBoxTracker
{
public:
	cv::KalmanFilter* kalman;
	double deltatime; //
	data LastResult;
	KalmanBoxTracker(data p,float dt=0.2,float Accel_noise_mag=0.5);
	~KalmanBoxTracker();
	data GetPrediction();
	data Update(data p, bool DataCorrect);
};

#endif
