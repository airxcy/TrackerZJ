#pragma once
#include "itf/trackers/KalmanBoxTracker.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>



KalmanBoxTracker::KalmanBoxTracker(data pt,float dt,float Accel_noise_mag)
{
    int state_dim = 8;
    int measure_dim = 4;
    LastResult = pt;
    LastResult.predict=false;
    deltatime = dt;
    /*we use 8-dim vector as the state, 4-dim vector as the measure vector, no control vector*/
    kalman = new cv::KalmanFilter(state_dim, measure_dim, 0);
    //transition matrix
    kalman->transitionMatrix = (cv::Mat_<float>(8, 8) << 1, 0, 0, 0, deltatime, 0, 0, 0,   0, 1, 0, 0, 0, deltatime, 0, 0,   0, 0, 1, 0, 0, 0, deltatime, 0,
                            0, 0, 0, 1, 0, 0, 0, deltatime,  0, 0, 0, 0, 1, 0, 0, 0,   0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0, 1); //system Matrix A
    //TODO: implement with the var--state_dim and measure_dim
    kalman->measurementMatrix = cv::Mat::eye(measure_dim, state_dim, CV_32F); //observation Matrix H


    kalman->measurementNoiseCov =25*cv::Mat::eye(measure_dim, measure_dim,CV_32F); // measurement noise R　
    kalman->errorCovPost =1000*cv::Mat::eye(state_dim, state_dim, CV_32F); //state covariance P　1000

    kalman->processNoiseCov = cv::Mat::eye(state_dim, state_dim, CV_32F); //process noise Q　

	
    kalman->statePre.at<float>(0) = 0.5*(pt.bbox[0]+pt.bbox[2]);  //x -center
    kalman->statePre.at<float>(1) = 0.5*(pt.bbox[1]+pt.bbox[3]);  //y -center
    kalman->statePre.at<float>(2) = pt.bbox[2]-pt.bbox[0];  // w
    kalman->statePre.at<float>(3) = pt.bbox[3]-pt.bbox[1];  // h
    kalman->statePre.at<float>(4) = 0; //Vx
    kalman->statePre.at<float>(5) = 0; //Vy
    kalman->statePre.at<float>(6) = 0; // Vw
    kalman->statePre.at<float>(7) = 0; // Vh

    kalman->statePost.at<float>(0) = 0.5*(pt.bbox[0]+pt.bbox[2]); //x -center
    kalman->statePost.at<float>(1) = 0.5*(pt.bbox[1]+pt.bbox[3]); //y -center
    kalman->statePost.at<float>(2) = pt.bbox[2]-pt.bbox[0]; //w
    kalman->statePost.at<float>(3) = pt.bbox[3]-pt.bbox[1]; //h
    kalman->statePost.at<float>(4) = 0; //Vx
    kalman->statePost.at<float>(5) = 0; //Vy
    kalman->statePost.at<float>(6) = 0; //Vw
    kalman->statePost.at<float>(7) = 0; //Vh


}

KalmanBoxTracker::~KalmanBoxTracker()
{
	delete kalman;
}
data KalmanBoxTracker::GetPrediction()
{

    cv::Mat prediction = kalman->predict();

     //cv::Point2d tmp=cv::Point2d(prediction.at<float>(0),prediction.at<float>(1));
     data tmpdata;
     double w = prediction.at<float>(2);
     double h = prediction.at<float>(3);
     tmpdata.bbox.push_back(prediction.at<float>(0) - 0.5*w); //x1
     tmpdata.bbox.push_back(prediction.at<float>(1) - 0.5*h); //y1
     tmpdata.bbox.push_back(prediction.at<float>(0) + 0.5*w); //x2
     tmpdata.bbox.push_back(prediction.at<float>(1) + 0.5*h); //y2
     tmpdata.score = -1; // flag
     tmpdata.index = -1; //flag 

     LastResult = tmpdata;
     return LastResult;
}

data KalmanBoxTracker::Update(data p, bool DataCorrect)
{

	cv::Mat measurement(4,1,CV_32FC1);
	if(!DataCorrect)
	{
		
	   measurement.at<float>(0) = 0.5*(LastResult.bbox[0]+LastResult.bbox[2]);  //update using prediction x-cen
	   measurement.at<float>(1) = 0.5*(LastResult.bbox[1]+LastResult.bbox[3]);  //y -cen
           measurement.at<float>(2) = (LastResult.bbox[2]-LastResult.bbox[0]); //w
           measurement.at<float>(3) = (LastResult.bbox[3]-LastResult.bbox[1]);//h
                
	}
	else
	{
	   #ifdef DEBUG
           //std::cout<<"p.point.x:"<<p.point.x<<", p.point.y:"<<p.point.y<<", p.w"<<p.w<<", p.h"<<p.h<<std::endl;
	   #endif
	   measurement.at<float>(0) = 0.5*(p.bbox[0]+p.bbox[2]);// p.point.x;  //update using measurements x -cen
	   measurement.at<float>(1) = 0.5*(p.bbox[1]+p.bbox[3]);//p.point.y; //y -cen
           measurement.at<float>(2) = (p.bbox[2]-p.bbox[0]); // w
           measurement.at<float>(3) = (p.bbox[3]-p.bbox[1]);//h
	}
	// Correction
	cv::Mat estimated = kalman->correct(measurement);

    data Tmpdata;

         double w=estimated.at<float>(2);
         double h=estimated.at<float>(3);
    Tmpdata.bbox.push_back(estimated.at<float>(0)-0.5*w);
         Tmpdata.bbox.push_back(estimated.at<float>(1)-0.5*h);
         Tmpdata.bbox.push_back(estimated.at<float>(0)+0.5*w);
         Tmpdata.bbox.push_back(estimated.at<float>(1)+0.5*h);
         Tmpdata.score=p.score;
         Tmpdata.index=p.index;
    Tmpdata.Feature=p.Feature;
    Tmpdata.hist_feature=p.hist_feature;
    if(!DataCorrect)
    {
        Tmpdata.predict=true;
    }
    else
        Tmpdata.predict=false;
         LastResult=Tmpdata;
	
	return LastResult;
}

