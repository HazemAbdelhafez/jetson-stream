/*
 * cpuRoutines.cpp
 *
 *  Created on: Dec 18, 2017
 *      Author: hazem
 */


#include "cpuRoutines.h"

void process_cblaskernel(Capsule *cap, KernelData* kernelData, u_int operationID){
	int n = cap->size;
	int vectorSize = n/2;
	item_t* x = &cap->input[0];
	item_t* y = &cap->input[vectorSize];			// We split the input array to two arrays for CBLAS routines

	switch(operationID){
		case SAXPY:
			// y = alpha*x + y
			cblas_daxpy(vectorSize,   2.0,  x, 1,   y, 1);
			cap->result = y[1];
			break;
		case DOT:
			// Dot product
			cblas_ddot(vectorSize, x,1, y, 1);
			cap->result = y[1];
			break;
		case EUCLIDEAN_DIST:
		{
			// Euclidean distance
			cblas_daxpy(vectorSize,   -1,  x, 1,   y, 1);
			double result = cblas_dnrm2(vectorSize, y, 1);
			cap->result = (item_t) result;
			break;
		}

		case MAT_VEC_MUL:
			// Matrix vector multiply, we initialise an auxilliary matrix to apply this operation.
			// y := alpha*A*x + beta*y , we assume square matrix for simplicity
			// Matrix size = vectorSize*vectorSize, Alpha is 2.3, Beta is 1.5. Just any two numbers
			// A is auxMatrix, LDA is max(1, vectorSize)
			cblas_dgemv(CblasRowMajor, CblasNoTrans, vectorSize, vectorSize, 2.3, kernelData->auxMatrix, std::max(1, vectorSize), x, 1, 1.5, y, 1);
			cap->result = (item_t) y[1];
			break;

		case MAT_MAT_MUL:
		{
			int M = 2;
			int K = n/M;
			int N = K;
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					M,		 								// M
					N,    									// N
					K,          									// K
					2.3,									// Alpha
					cap->input, 	   						// Matrix A
					K,		 								// LDA
					kernelData->auxMatrix,  			// Matrix B
					K,	 									// LDB
					1.5,									// Beta
					kernelData->outputMatrix,			// Matrix C
					N);										// LDC
		}
		cap->result = (item_t) y[1];
		break;

		default:
			// y = alpha*x + y
			cblas_daxpy(vectorSize,   2.0,  x, 1,   y, 1);
			cap->result = y[1];
			break;
	}

}

void process_coldstart_cblaskernel(Capsule *cap, KernelData* kernelData, u_int operationID){
	int n = cap->size;
	int vectorSize = 8;
	item_t* x = &cap->input[0];
	item_t* y = &cap->input[vectorSize];			// We split the input array to two arrays for CBLAS routines
	// y = alpha*x + y
	cblas_daxpy(vectorSize,   2.0,  x, 1,   y, 1);
	cap->result = y[1];
}

void *workerRoutine (void *args) {

	// Socket to talk to dispatcher (Messages are shared in-memory)
	ThreadArgs* threadArgs = ((NotifierArgs*) args)->threadArgs;

	// Special cases. Check the switch statement in the main calling method to know more.
	if (threadArgs->singleThreaded){
		threadArgs->numThreads = 1;
		// Set number of threads used by blas
		openblas_set_num_threads(threadArgs->numBLASThreads);
	} else {
		// Set number of threads used by blas
		openblas_set_num_threads(1);
	}

	void *receiver = zmq_socket (threadArgs->context, ZMQ_PULL);

	u_int maxOutstandingMsgs = MAX_PENDING_MSGS;

	zmq_setsockopt(receiver, ZMQ_RCVHWM, (void*) &maxOutstandingMsgs, sizeof(maxOutstandingMsgs));
	zmq_setsockopt(receiver, ZMQ_SNDHWM, (void*) &maxOutstandingMsgs, sizeof(maxOutstandingMsgs));
	zmq_connect (receiver, WORKERS_ENDPOINT);

	std::vector<timespec> recvTimes;
	recvTimes.reserve(MSG_STATISTICS_BUFFER_SIZE);

	std::vector<timespec> procTimes;
	procTimes.reserve(MSG_STATISTICS_BUFFER_SIZE);

	int numMsgs = 0;

	// Register interrupt handler for clean exit.
	s_catch_signals ();

	u_int msgSize  =  threadArgs->msgSize;
	u_int numOps = threadArgs->numOps;
	Capsule cap;
	cap.size = msgSize;

	timespec recvTime;
	timespec procTime;

	zmsg msg;
	int rc =zmq_msg_init_size(&msg, msgSize*sizeof(item_t));
	assert(rc==0);

	// Init kernel helper data
	KernelData kernelData;
	int matDim = msgSize/2;

	// Init the auxilliary matrix that we will use in the computation kernel
	kernelData.auxMatrix = (item_t*) malloc(matDim*matDim*sizeof(item_t));

	// Init all matrix elements with specific value
	item_t randomValue = 3.5;
	fill_2d_matrix(kernelData.auxMatrix, matDim, matDim, randomValue);

	// Special case: We need an output matrix too.
	if (threadArgs->numOps == MAT_MAT_MUL){
		int rows = 2;
		kernelData.outputMatrix = (item_t*) malloc(rows*matDim*sizeof(item_t));
		fill_2d_matrix(kernelData.outputMatrix, rows, matDim, 0);
	}

	// Timing info
	timespec testStartTime;
	timespec currentTime = {0, 0}; 										// Test cut-off time in seconds

	// Set scaling governer to userspace
	if (threadArgs->controlJetson){
		jetson::init();
		jetson::setToMax();
	}

	// Mark thread as ready
	std::atomic<u_int>* isReady = threadArgs->threadsReady;
	u_int threadId = (*isReady)++;

//	int coldStartSamples = (int) threadArgs->coldStartIgnoredSamples;
//
//	// Before the cold start samples
//	while (!s_interrupted && numMsgs < coldStartSamples) {
//		// Receive message
//		rc = zmq_msg_recv(&msg, receiver, 0);
//		if (rc == -1)
//			break;
//		cap.input = (item_t*) zmq_msg_data(&msg);
//		timer::markTime(&recvTime);
//
//		//	Process message data
//		process_coldstart_cblaskernel(&cap, &kernelData,numOps);
//		timer::markTime(&procTime);
//
//		//  Record the time stamps.
//		recvTimes.push_back(recvTime);
//		procTimes.push_back(procTime);
//		numMsgs++;
//		timer::markTime(&currentTime);
//	}

	// After the cold start samples
	if (threadArgs->controlJetson){
		// Add timing information to stop the experiments once reached.
		timer::markTime(&testStartTime);
		timespec testStopTime = timer::addTime(testStartTime, threadArgs->testTime, 0);
		while (!s_interrupted && timer::getDuration(currentTime, testStopTime) > 0) {
			// Receive message
			rc = zmq_msg_recv(&msg, receiver, 0);

			cap.input = (item_t*) zmq_msg_data(&msg);
			timer::markTime(&recvTime);

			// Set the hardware to maximum settings
			jetson::setToMax();

			//	Process message data
			process_cblaskernel(&cap, &kernelData,numOps);
			timer::markTime(&procTime);

			// Set the hardware to minimum settings
			jetson::setToDefault();

			//  Record the time stamps.
			recvTimes.push_back(recvTime);
			procTimes.push_back(procTime);

			numMsgs++;
			timer::markTime(&currentTime);
		}
	} else{
		// Add timing information to stop the experiments once reached.
		timer::markTime(&testStartTime);
		timespec testStopTime = timer::addTime(testStartTime, threadArgs->testTime, 0);
		while (!s_interrupted && timer::getDuration(currentTime, testStopTime) > 0) {
			// Receive message
			rc = zmq_msg_recv(&msg, receiver, 0);

			cap.input = (item_t*) zmq_msg_data(&msg);
			timer::markTime(&recvTime);

			//	Process message data
			process_cblaskernel(&cap, &kernelData,numOps);
			timer::markTime(&procTime);

			//  Record the time stamps.
			recvTimes.push_back(recvTime);
			procTimes.push_back(procTime);

			numMsgs++;
			timer::markTime(&currentTime);
		}
	}
	if (recvTimes.size() == 0 || procTimes.size() == 0){
		std::cout << "[Error] No output metrics. Aborting.";
		// Send producer to stop sending more messages and exit.
		pthread_t stopNotifier;
		pthread_create(&stopNotifier, NULL, stopProducer, args);
		pthread_join(stopNotifier, NULL);
		zmq_ctx_destroy (threadArgs->context);
		return NULL;
	}
	if (threadArgs->verbose)
		std::cout << "Collecting metrics " << std::endl;

	// Collect metrics before exit.
	metrics::collectMetrics(threadArgs, threadId, numMsgs, recvTimes, procTimes);

	zmq_msg_close(&msg);
	zmq_close (receiver);

	// Announce that I am done
	std::atomic<u_int>* isDone = threadArgs->threadsDone;
	(*isDone)++;

	std::atomic<u_int>* collectedMetrics = threadArgs->collectedMetrics;

	// Report the metrics if you are the last thread to finish
	if ((*isDone) == threadArgs->numThreads){
		// Send producer to stop sending more messages and exit.
		pthread_t stopNotifier;
		pthread_create(&stopNotifier, NULL, stopProducer, args);
		pthread_join(stopNotifier, NULL);

		metrics::report(threadArgs);
		(*collectedMetrics)++;

		if (threadArgs->controlJetson)
			jetson::setToMax();
		// A not very neat way to close the application once we finished processing all messages
		zmq_ctx_destroy (threadArgs->context);
	}
	// Keep all threads alive until the main thread is done collecting metrics. Because else, the recvTimes and procTimes are null.
	while ((*collectedMetrics) != 1){
		usleep(500);
		continue;
	}
	return NULL;
}

void* startProducer(void* args){
	NotifierArgs* notifierArgs = (NotifierArgs*) args;
	void* sender  = zmq_socket(notifierArgs->threadArgs->context, ZMQ_PAIR);
	int rc = zmq_connect(sender, getURL(TCP_PROTOC, *(notifierArgs->producerAddress), notifierArgs->notifierPort).c_str());
	if(rc == -1){
		std::cerr << "[Error] Failed to bind to the specified address for the producer start notifier" << std::endl;
		std::cerr << "Error code is: " << errno << std::endl;
	}
	std::atomic<u_int>* isReady = notifierArgs->threadArgs->threadsReady;

	// Register interrupt handler for clean exit.
	s_catch_signals ();
	while(!s_interrupted && *isReady < notifierArgs->threadArgs->numThreads){
		sleep(1);
		continue;
	}
	rc = zmq_send(sender, "start", 5, 0);
	assert(rc ==5);
	zmq_close(sender);
	if (notifierArgs->threadArgs->verbose){
		printf("[Info] Notified the producer to start sending messages.\n");
		timespec notifyTime;
		timer::markTime(&notifyTime);
		timer::printTime(notifyTime, "[Info] Notify time");
	}
	return NULL;
}


void*  stopProducer(void* args){
	NotifierArgs* notifierArgs = (NotifierArgs*) args;
	void* sender  = zmq_socket(notifierArgs->threadArgs->context, ZMQ_REQ);
	int rc = zmq_connect(sender, getURL(TCP_PROTOC, *(notifierArgs->producerAddress), notifierArgs->notifierPort+5).c_str());
	if(rc == -1){
		std::cerr << "[Error] Failed to bind to the specified address for the producer stop notifier" << std::endl;
		std::cerr << "Error code is: " << errno << std::endl;
	}
	rc = zmq_send(sender, "start", 5, 0);
	assert(rc ==5);
	char recv[5];
	zmq_recv(sender, recv, 5, 0);
	zmq_close(sender);
	if (notifierArgs->threadArgs->verbose)
		printf("[Info] Notified the producer to stop sending messages.\n");
	return NULL;
}
