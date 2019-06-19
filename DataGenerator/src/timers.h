/*
 * timers.h
 *
 *  Created on: Dec 15, 2017
 *      Author: hazem
 */

#ifndef TIMERS_H_
#define TIMERS_H_

#define BILLION 1E9

#include "time.h"
#include <string>
namespace timer{
void markStart(timespec *start);
void markStop(timespec *stop);
void markTime(timespec *time);
timespec addTime(timespec time1, long int t_secs, long int t_nsecs);
void printTime(timespec time, std::string msg);
double duration(timespec start, timespec stop);
double getDuration(timespec start, timespec stop);
}
#endif /* TIMERS_H_ */
