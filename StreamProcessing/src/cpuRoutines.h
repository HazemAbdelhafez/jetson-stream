/*
 * cpuRoutines.h
 *
 *  Created on: Dec 18, 2017
 *      Author: hazem
 */

#ifndef CPUROUTINES_H_
#define CPUROUTINES_H_

#include "commons.h"
#include "jetson-controller.h"

void *workerRoutine (void *cxt);
void *pKernelDriver (void *cxt);
void *bKernelDriver (void *cxt);
void* startProducer(void* cxt);
void*  stopProducer(void* cxt);
#endif /* CPUROUTINES_H_ */
