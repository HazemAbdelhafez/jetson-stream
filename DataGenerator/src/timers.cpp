/*
 * timers.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: hazem
 */

#include "timers.h"

#include <iostream>
#include <iomanip>
#include <cmath>

namespace timer{
void markStart(timespec *start){
	clock_gettime(CLOCK_MONOTONIC, start);
}

void markStop(timespec *stop){
	clock_gettime(CLOCK_MONOTONIC, stop);
}
void markTime(timespec *time){
	clock_gettime(CLOCK_MONOTONIC, time);
}
double duration(timespec start, timespec stop){
	return getDuration(start, stop);
}

inline double toNanoseconds(struct timespec* ts) {
    return (double) (ts->tv_sec * (double)1000000000L + ts->tv_nsec);
}

timespec addTime(timespec time1, long int t_secs, long int t_nsecs){
	timespec result;
	double newTime = toNanoseconds(&time1) + (double) (t_secs*BILLION + (double) t_nsecs);
	result.tv_sec =  (long int) floor(newTime/BILLION);
	result.tv_nsec = (long int) (newTime - floor(newTime/BILLION)*BILLION);
	return result;
}

void printTime(timespec time, std::string msg){
	double seconds = (double) time.tv_sec;
	double nsecs = (double) time.tv_nsec/BILLION;
	std::cout << std::setprecision(10)<< msg << " time is: " <<  seconds + nsecs << std::endl;
}
double getDuration(timespec start, timespec stop){

	double duration = 0;
	double first = (double) start.tv_sec +  (double) start.tv_nsec/BILLION;
	double last = (double) stop.tv_sec +  (double) stop.tv_nsec/BILLION;
	duration  = last - first;
	return duration;
}
}
