/*
 * Commons.h
 *
 *  Created on: Dec 18, 2017
 *      Author: hazem
 */

#ifndef COMMONS_H_
#define COMMONS_H_
#include "zmq.h"
#include <stdlib.h>
#include <iostream>
#include <string>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "config.h"
#include <string.h>
#include "time.h"
#include <string>
#include <signal.h>
#include <errno.h>
#include <atomic>
#include "cxxopts.hpp"
#include <fstream>
#include <iomanip>
#include "cblas.h"
#include "math.h"
#define BILLION 1000000000L



typedef zmq_msg_t zmsg;
typedef double item_t;
struct Capsule{
	item_t* input;
	int size;
	item_t result;
};

struct ThreadArgs{
	void* context;
	std::atomic<u_int>* threadsReady;
	std::atomic<u_int>* threadsDone;
	std::atomic<u_int>* collectedMetrics;
	u_int numOps;
	u_int msgSize;
	u_int numMsgs;
	u_int numThreads;
	double* throughputResults;
	double* ProcLatencyResults;
	double* totalLatencyResults;
	double* maxProcLatency;
	double* minProcLatency;
	double* maxTotalLatency;
	double* minTotalLatency;
	double* sumProcLatency;
	double* sumTotalLatency;
	std::vector<timespec*> recvTimes;
	std::vector<timespec*> procTimes;
	u_int* numProcessedMsgs;
	u_int verbose;
	std::string* outputFilePath;
	u_int coldStartIgnoredSamples;
	long int  testTime;
	u_int computeMode;
	u_int gpuBatchSize;
	u_int numBLASThreads;
	bool singleThreaded;
	bool controlJetson;
};

struct GPUThreadArgs{
	ThreadArgs* threadArgs;
	u_int batchSize;
	u_int threadsPerBlock;
};

struct NotifierArgs{
	ThreadArgs* threadArgs;
	std::string* producerAddress;
	u_int notifierPort;
	u_int numWorkers;
};

struct KernelData{
	item_t* auxMatrix;
	item_t* outputMatrix;
};

struct Stats{
	double procLatencySD;
	double totalLatencySD;
	double avgProcLatency;
	double avgTotalLatency;
};

namespace timer{
	void markStart(timespec *start);
	void markStop(timespec *stop);
	void markTime(timespec *time);
	double duration(timespec start, timespec stop);
	double getDuration(timespec start, timespec stop);
	timespec addTime(timespec time1, long int t_secs, long int t_nsecs);
	void printTime(timespec time, std::string msg);
	void printTime(timespec time, std::string msg);
	inline double toNanoseconds(struct timespec* ts) {
	    return (double) (ts->tv_sec * (double)1000000000L + ts->tv_nsec);
	}
}

namespace metrics{
	double maxProcLatency(std::vector<timespec> recvTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples);
	double minProcLatency(std::vector<timespec> recvTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples);
	double maxTotalLatency(std::vector<timespec> receiveTime,  u_int m, u_int ignoredSamples);
	double minTotalLatency(std::vector<timespec> receiveTime,  u_int m, u_int ignoredSamples);

	double avgRecvThroughput(std::vector<timespec> startTime, std::vector<timespec> stopTime, u_int m, u_int ignoredSamples);
	double avgProcThroughput(std::vector<timespec> startTime, std::vector<timespec> stopTime, u_int m, u_int ignoredSamples);
	double avgProcLatency(std::vector<timespec> startTime, std::vector<timespec> stopTime, u_int m, u_int ignoredSamples);
	double avgTotalLatency(std::vector<timespec> startTime, std::vector<timespec> stopTime, u_int m, u_int ignoredSamples);

	double sumProcLatency(std::vector<timespec> recvTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples);
	double sumTotalLatency(std::vector<timespec> recvTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples);

	double getAverage(double* data, u_int n);
	double getMax(double* data, u_int n);
	double getMin(double* data, u_int n);

	void getSD(ThreadArgs* args, Stats* stats);
	void getAverage(ThreadArgs* args, Stats* stats);
	void collectMetrics(ThreadArgs* threadArgs, int threadId, int numMsgs, std::vector<timespec> &recvTimes, std::vector<timespec> &procTimes);
	template <typename T>
	T accumlate(T* data, u_int n){
		T sum = 0;
		for (u_int i=0; i<n; i++)
			sum += data[i];
		return sum;
	}
	void report(ThreadArgs* threadArgs);
}

/** [CPU] FLOPs: (Scalar and Vector) -> Add, Subtract, Multiply, FMA, Division
 * for now we will use the scalar addition only */

__host__ __device__ void process(Capsule *cap, u_int numOps);
__host__ __device__ void process_cblaskernel(Capsule *cap, KernelData* auxData, u_int numOps);

void fill_2d_matrix(item_t* matrix, int n, int m, item_t value) ;
void printArray(int* data, unsigned int n, const char* msg);
bool checkResults(Capsule *input, int m, u_int numOps);
std::string getURL();
std::string getURL(std::string proto, std::string address, u_int port);

static int s_interrupted = 0;
static inline void s_signal_handler (int signal_value)
{
	std::cout << "[Warning] Caught signal" <<std::endl;
    s_interrupted = 1;
}
//static int verbose = 0;
//static inline void setVerbose(int value){
//	verbose = value;
//}
static void s_catch_signals (void)
{
    struct sigaction action;
    action.sa_handler = s_signal_handler;
    action.sa_flags = 0;
    sigemptyset (&action.sa_mask);
    sigaction (SIGINT, &action, NULL);
    sigaction (SIGTERM, &action, NULL);
    sigaction (SIGABRT, &action, NULL);
}
#endif /* COMMONS_H_ */
