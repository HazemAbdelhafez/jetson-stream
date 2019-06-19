/*
 * commons.cpp
 *
 *  Created on: Dec 20, 2017
 *      Author: hazem
 */

#include "commons.h"




std::string getURL(){
	std::string url = UDP_PROTOC;
	url.append("://");
	url.append(DEFAULT_SERVER_ADDRESS);
	url.append(":");
	url.append(DEFAULT_DATA_PORT);
//	std::cout << "[Server] Binding URL: " << url << std::endl;
	return url;
}

std::string getURL(std::string proto, std::string address, u_int port){
	std::string url = proto;
	url.append("://");
	url.append(address);
	url.append(":");
	url.append(std::to_string(port));
//    std::cout << "[Server] Binding URL: " << url << std::endl;
	return url;
}

void printArray(int* data, unsigned int n, const char* msg){
	std::cout << msg << ": [";
	std::cout << data[0];
	if (n==1)
		return;
	for(unsigned int i=1; i<n;i++){
		std::cout << ", " << data[i];
	}
	std::cout << "]" <<std::endl;

}

/** Check the correctness of computed results on two different arrays of inputs. */
bool checkResults(Capsule *cap, int m, u_int numOps){
	int diff = 0;
	int n = cap[0].size;
	for(int k=0; k<m; k++){
		unsigned int x = 0;
		for(unsigned int i = 0; i< numOps/2; i++){
			for (int j=0; j<n; j++)
				x += i + cap[k].input[j]; // Two FLOPs
		}
		diff += cap[k].result - x;
	}
	return (diff == 0);
}



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
	double duration = getDuration(start, stop);
//	std::cout << "[Info] Duration is: " << duration << std::endl;
	return duration;
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

namespace metrics{
	double maxProcLatency(std::vector<timespec> recvTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples){
		double maxLatency = 0;
		// We start at (ignoredSamples - 1) instead of ignoredSamples to have data for the last -1  sample to process its processing time compared to the receive time of the last sample
		for (int i=ignoredSamples-1; i < m-1; i++){
			double latency = timer::getDuration(recvTime[i], procTime[i]);
			if (latency > maxLatency)
				maxLatency = latency;
		}
		return maxLatency;
	}
	double minProcLatency(std::vector<timespec> recvTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples){
		double minLatency = std::numeric_limits<double>::max();
		for (int i=ignoredSamples-1; i < m-1; i++){
			double latency = timer::getDuration(recvTime[i], procTime[i]);
			if (latency < minLatency)
				minLatency = latency;
		}
		return minLatency;
	}
	double maxTotalLatency(std::vector<timespec> receiveTime,  u_int m, u_int ignoredSamples){
		double maxLatency = 0;
		int count = 0;
		for (int i = ignoredSamples-1; i < m-1; i++){
			double latency = timer::getDuration(receiveTime[i], receiveTime[i+1]);
			if (latency > maxLatency)
				maxLatency = latency;
		}
		return maxLatency;
	}
	double minTotalLatency(std::vector<timespec> receiveTime, u_int m, u_int ignoredSamples){
		double minLatency =  std::numeric_limits<double>::max();
		for (int i = ignoredSamples-1; i < m-1; i++){
			double latency = timer::getDuration(receiveTime[i], receiveTime[i+1]);
			if (latency < minLatency)
				minLatency = latency;
		}
		return minLatency;
	}


	// Vector based methods
	double avgProcThroughput(std::vector<timespec> recvTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples){
		double throughput = 0;
		/** PTn = 1/Dn, Dn = Pn - Pn-1, PTn is processing throughput at message n,
		 * Dn is the difference between the time Pn at which message n was processed,
		 * and time Pn-1 at which message n-1 was processed. */
		int numMsgs = m-ignoredSamples;
		throughput = (double) numMsgs/(timer::getDuration(recvTime[ignoredSamples-1],procTime[m-2]));
		return throughput;
	}
	double avgTotalLatency(std::vector<timespec> receiveTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples){
		/** L = Rn - Rn-1, Ln is message end-to-end latency for message n-1,
		 * Rn is the time at which message n was received,
		 * Rn-1 is the time at which message n-1 was received. */
		int numMsgs = m-ignoredSamples;
		// Here we start exceptionally from igrnoredSamples-1 not ignoredSamples  to include the latency for the first received message instead of ignoring it and dividing by (m-1), I believe this is more accuarate.
		double latency = timer::getDuration(receiveTime[ignoredSamples - 1], receiveTime[m-1]);
		return (double) (latency/numMsgs);
	}
	double avgProcLatency(std::vector<timespec> receiveTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples){
		double latency = 0;
		/** Ln = Pn - Rn, Ln is message processing latency for message n,
		 * Pn is the time at which message n is completely processed,
		 * Rn is the time at which message n was received. */
		double diff = 0;
		int numMsgs = m-ignoredSamples;
		for (int i=ignoredSamples-1; i < m-1; i++){
			diff += timer::getDuration(receiveTime[i], procTime[i]);
		}
		latency = (double) diff/numMsgs;
		return latency;
	}

	double sumProcLatency(std::vector<timespec> receiveTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples){
		double sum = 0;
		int numMsgs = m-ignoredSamples;
		for (int i=ignoredSamples-1; i < m-1; i++){
			sum += timer::getDuration(receiveTime[i], procTime[i]);
		}
		return sum;
	}
	double sumTotalLatency(std::vector<timespec> receiveTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples){

		int numMsgs = m-ignoredSamples;
		// Here we start exceptionally from igrnoredSamples-1 not ignoredSamples  to include the latency for the first received message instead of ignoring it and dividing by (m-1), I believe this is more accuarate.
		double sum = timer::getDuration(receiveTime[ignoredSamples - 1], receiveTime[m-1]);
		return sum;
	}

	double getAverage(double* data, u_int n){
		double average = 0;
		for (u_int i=0; i<n; i++)
			average += data[i];
		average = average/n;
		return average;
	}

	double getProcLatencyVariance(std::vector<timespec> receiveTime, std::vector<timespec> procTime, u_int m, u_int ignoredSamples, double average){
		double variance = 0;
		double diff = 0;
		int numMsgs = m-ignoredSamples;
		for (int i=ignoredSamples-1; i < m-1; i++){
			diff = timer::getDuration(receiveTime[i], procTime[i]) - average;
			variance += pow(diff, 2);
		}
		return variance;
	}

	double getProcLatencyVariance(timespec* receiveTime, timespec* procTime, u_int m, u_int ignoredSamples, double average){
		double variance = 0;
		double diff = 0;
		int numMsgs = m-ignoredSamples;
		for (int i=ignoredSamples-1; i < m-1; i++){
			diff = timer::getDuration(receiveTime[i], procTime[i]) - average;
			variance += pow(diff, 2);
		}
		return variance;
	}

	double getTotalLatencyVariance(timespec* receiveTime, timespec* procTime, u_int m, u_int ignoredSamples, double average){
		double variance = 0;
		double diff = 0;

		double test = 0;
		// Here we start exceptionally from igrnoredSamples-1 not ignoredSamples  to include the latency for the first received message instead of ignoring it and dividing by (m-1), I believe this is more accuarate.
		for (int i=ignoredSamples-1; i < m-1; i++){
			test += timer::getDuration(receiveTime[i], receiveTime[i+1]);
			diff = timer::getDuration(receiveTime[i], receiveTime[i+1]) - average;
			variance += pow(diff, 2);
		}
		return variance;
	}

	void getAverage(ThreadArgs* threadArgs, Stats* stats){
		// Init
		stats->avgProcLatency = 0;
		stats->avgTotalLatency = 0;

		double sumProcLatency = 0;
		double sumTotalLatency = 0;
		unsigned long int  totalNumMsgs = 0;

		for (int k=0; k < threadArgs->numThreads; k++){
			sumProcLatency += threadArgs->sumProcLatency[k];
			sumTotalLatency += threadArgs->sumTotalLatency[k];
			totalNumMsgs += (threadArgs->numProcessedMsgs[k] - threadArgs->coldStartIgnoredSamples);   // Check the avg, sum methods to know why I subtracted the cold samples.
		}
		stats->avgProcLatency = (double) sumProcLatency/totalNumMsgs;
		stats->avgTotalLatency = (double) sumTotalLatency/totalNumMsgs;
		if (threadArgs->verbose){
			std::cout << "Sum of proc latency: " << sumProcLatency << std::endl;
			std::cout << "Sum of total latency: " << sumTotalLatency << std::endl;
			std::cout << "Sum of number of msgs: " << totalNumMsgs << std::endl;
			std::cout << "Average proc latency: " << stats->avgProcLatency << std::endl;
			std::cout << "Average total latency: " << stats->avgTotalLatency << std::endl;
		}
	}
	void getSD(ThreadArgs* threadArgs, Stats* stats){
		// Init
		stats->procLatencySD = 0;
		stats->totalLatencySD = 0;

		double procLatencyVar = 0;
		double totalLatencyVar = 0;
		unsigned long int  totalNumMsgs = 0;

		int numThreads = (int) threadArgs->numThreads;

		for (int k=0; k < numThreads; k++){
			timespec* receiveTimes = threadArgs->recvTimes[k];
			timespec* procTimes = threadArgs->procTimes[k];
			procLatencyVar +=getProcLatencyVariance(receiveTimes, procTimes, threadArgs->numProcessedMsgs[k], threadArgs->coldStartIgnoredSamples, stats->avgProcLatency) ;
			totalLatencyVar += getTotalLatencyVariance(receiveTimes, procTimes, threadArgs->numProcessedMsgs[k], threadArgs->coldStartIgnoredSamples, stats->avgTotalLatency) ;
			totalNumMsgs += (threadArgs->numProcessedMsgs[k] - threadArgs->coldStartIgnoredSamples);   // Check the avg, sum methods to know why I subtracted the cold samples.
		}
		stats->procLatencySD = sqrt(procLatencyVar/totalNumMsgs);
		stats->totalLatencySD = sqrt(totalLatencyVar/totalNumMsgs);
		if (threadArgs->verbose){
			std::cout << "Var of proc latency: " << procLatencyVar << std::endl;
			std::cout << "Var of total latency: " << totalLatencyVar << std::endl;
			std::cout << "Sum of number of msgs: " << totalNumMsgs << std::endl;
			std::cout << "SD proc latency: " << stats->procLatencySD << std::endl;
			std::cout << "SD total latency: " << stats->totalLatencySD << std::endl;
		}
	}

	double getMax(double* data, u_int n){
		double max = 0;
		for (u_int i=0; i<n; i++)
			if (data[i] > max)
				max = data[i];
		return max;
	}
	double getMin(double* data, u_int n){
		double min =  std::numeric_limits<double>::max();
		for (u_int i=0; i<n; i++)
			if (data[i] < min)
				min = data[i];
		return min;
	}
	void report(ThreadArgs* threadArgs){
	    // Report the results
		u_int numThreads = (u_int) *(threadArgs->threadsReady);
		if (threadArgs->verbose){
			std::cout << "[Info] Total number of received messages is " << metrics::accumlate(threadArgs->numProcessedMsgs, numThreads) << std::endl;
			std::cout << "[Info] Total number of active threads is " << numThreads << std::endl;
			std::cout << "[Info] Overall system throughput is " << metrics::accumlate(threadArgs->throughputResults, numThreads) << " messages/second"<< std::endl;
			std::cout << "[Info] Average message processing latency per thread is " << metrics::getAverage(threadArgs->ProcLatencyResults, numThreads) << " seconds" << std::endl;
			std::cout << "[Info] Average message total latency per thread is " << metrics::getAverage(threadArgs->totalLatencyResults, numThreads) << " seconds" << std::endl;
			std::cout << "[Info] Average thread processing throughput is " << metrics::getAverage(threadArgs->throughputResults, numThreads) << " messages/second" << std::endl;
		}
		Stats stats;
		metrics::getAverage(threadArgs, &stats);
		metrics::getSD(threadArgs, &stats);

		// Write the results to a file so that the run script can read it, the precision is set to the highest
		std::ofstream outputFile;
		outputFile.open(*(threadArgs->outputFilePath), std::ios_base::app);
		outputFile
		                << threadArgs->numMsgs<< " "
		                << threadArgs->msgSize<< " "
		                << threadArgs->numOps << " "
		                << threadArgs->gpuBatchSize << " "
		                << threadArgs->numBLASThreads << " "
				<< metrics::accumlate(threadArgs->numProcessedMsgs, numThreads) << " "
				<< metrics::accumlate(threadArgs->throughputResults, numThreads) << " "
				<< stats.avgTotalLatency << " "
				<< stats.avgProcLatency << " "
				<< metrics::getAverage(threadArgs->throughputResults, numThreads) << " "
				<< getMax(threadArgs->maxTotalLatency, numThreads) << " "
				<< getMin(threadArgs->minTotalLatency, numThreads) << " "
				<< getMax(threadArgs->maxProcLatency, numThreads) << " "
				<< getMin(threadArgs->minProcLatency, numThreads) << " "
				<< stats.totalLatencySD << " "
				<< stats.procLatencySD << " "
				<< std::endl;
		outputFile.close();
	}

	void collectMetrics(ThreadArgs* threadArgs, int threadId, int numMsgs, std::vector<timespec> &recvTimes, std::vector<timespec> &procTimes){

		threadArgs->recvTimes[threadId] = recvTimes.data();
		threadArgs->procTimes[threadId] = procTimes.data();

		threadArgs->ProcLatencyResults[threadId] = metrics::avgProcLatency(recvTimes, procTimes, numMsgs, threadArgs->coldStartIgnoredSamples);
		threadArgs->totalLatencyResults[threadId] = metrics::avgTotalLatency(recvTimes, procTimes, numMsgs, threadArgs->coldStartIgnoredSamples);

		threadArgs->sumProcLatency[threadId] = metrics::sumProcLatency(recvTimes, procTimes, numMsgs, threadArgs->coldStartIgnoredSamples);
		threadArgs->sumTotalLatency[threadId] = metrics::sumTotalLatency(recvTimes, procTimes, numMsgs, threadArgs->coldStartIgnoredSamples);

		threadArgs->throughputResults[threadId] = metrics::avgProcThroughput(recvTimes, procTimes, numMsgs, threadArgs->coldStartIgnoredSamples);
		threadArgs->numProcessedMsgs[threadId] = numMsgs;

		threadArgs->maxTotalLatency[threadId] = metrics::maxTotalLatency(recvTimes, numMsgs, threadArgs->coldStartIgnoredSamples);
		threadArgs->maxProcLatency[threadId] = metrics::maxProcLatency(recvTimes, procTimes, numMsgs, threadArgs->coldStartIgnoredSamples);

		threadArgs->minTotalLatency[threadId] = metrics::minTotalLatency(recvTimes, numMsgs, threadArgs->coldStartIgnoredSamples);
		threadArgs->minProcLatency[threadId] = metrics::minProcLatency(recvTimes, procTimes, numMsgs, threadArgs->coldStartIgnoredSamples);

		if (threadArgs->verbose){
			std::cout << "[Thread-" << threadId << "] Average latency is: " << threadArgs->ProcLatencyResults[threadId] << std::endl;
			std::cout << "[Thread-" << threadId << "] Number of messages is: " << threadArgs->numProcessedMsgs[threadId] << std::endl;
			std::cout << "First sample at: " << recvTimes[0].tv_sec << std::endl;
			std::cout << "Last sample at: " << recvTimes[numMsgs-1].tv_sec << std::endl;
		}
	}
}

void fill_2d_matrix(item_t* matrix, int n, int m, item_t value) {
	int i, j;
	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++)
			matrix[i*m+j] = value;
}

