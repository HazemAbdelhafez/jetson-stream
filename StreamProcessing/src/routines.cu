/*
 * routines.cu
 *
 *  Created on: Dec 19, 2017
 *      Author: hazem
 */



#include "commons.h"

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/** [CPU] FLOPs: (Scalar and Vector) -> Add, Subtract, Multiply, FMA, Division
 * for now we will use the scalar addition only */

__host__ __device__ void process(Capsule *cap, u_int numOps){
	item_t x = 0;
	int n = cap->size;
	for(unsigned int i = 0; i < numOps/2; i++){
		for (int j=0; j<n; j++)
			x += i + cap->input[j]; // Two FLOPs
	}
	cap->result = x;
}

/** Launch computation on multiple-messages as a batch. */
__global__ void bKernel(Capsule* __restrict__ cap, u_int numOps, int batchSize){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	// Each thread will handle a message
	if (idx < batchSize)
		process(&cap[idx], numOps);
}

void initBuffer(Capsule *buffer, int size, int msgSize){
	for (int i = 0; i<size; i++){
		buffer[i].result = 0;
		buffer[i].size = msgSize;

		// This separate allocation of each input array might not be optimal for memory access
		// coalescing on the GPU. TODO: Change this to maybe a column major matrix and check if
		// it makes any difference in the processing time.
		cudaMallocManaged((void**) &buffer[i].input, msgSize*sizeof(int), cudaMemAttachGlobal);
	}
}

void *bKernelDriver (void *args){

	//  Socket to talk to dispatcher
	GPUThreadArgs* gpuThreadArgs = (GPUThreadArgs*) args;
    void *receiver = zmq_socket (gpuThreadArgs->threadArgs->context, ZMQ_PULL);
    zmq_connect (receiver, WORKERS_ENDPOINT);

    // Prepare messages container.
    size_t msgBytes = (gpuThreadArgs->threadArgs->msgSize)*sizeof(int);
	zmsg msg;
	int rc = zmq_msg_init_size(&msg, msgBytes);
	assert(rc == 0);

	// Kernel launch parameters.
	int batchSize = (int) gpuThreadArgs->batchSize;
    int threadsPerBlock = gpuThreadArgs->threadsPerBlock;
    int blocksPerGrid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    u_int numOps = gpuThreadArgs->threadArgs->numOps;

	int pendingMsgs = 0;
	Capsule *buffer;
	cudaMallocManaged((void**) &buffer, batchSize*sizeof(Capsule), cudaMemAttachGlobal);
	initBuffer(buffer, batchSize, msgBytes/sizeof(int));


	// Initialize the timing results' arrays
	u_int numMsgs = gpuThreadArgs->threadArgs->numMsgs;
	timespec recvTimes[numMsgs];
	timespec procTimes[numMsgs];

	int startIndex = 0;
	int stopIndex = 0;

	// Register interrupt handler for clean exit.
	s_catch_signals ();

	// Mark this thread as ready
	std::atomic<u_int>* isReady = gpuThreadArgs->threadArgs->threadsReady;
	u_int threadId = (*isReady)++;

	while (!s_interrupted && startIndex < numMsgs) {

		// Receive message
		rc = zmq_msg_recv(&msg, receiver, 0);
		if (rc == -1){
			break;
		}
		int* data = (int*) zmq_msg_data(&msg);
		timer::markTime(&recvTimes[startIndex]);

		// Accumulate messages in a buffer. TODO: Accumulate messages first in Malloc buffer then copy as whole to the GPU instead of copying message by message which is inefficient.
		HANDLE_ERROR(cudaMemcpy(buffer[pendingMsgs].input, data, msgBytes, cudaMemcpyHostToDevice));
		pendingMsgs++;

		if (pendingMsgs == batchSize){

			// Launch kernel
			bKernel<<<blocksPerGrid, threadsPerBlock>>>(buffer, numOps, batchSize);
			cudaStreamSynchronize(0);
//			printf("Here\n");
			// Mark stop time of "BatchSize" messages at once.
			for (int i=0; i<batchSize; i++)
				timer::markTime(&procTimes[stopIndex*batchSize + i]);
			// Reset counter
			pendingMsgs = 0;
			stopIndex++;
//			bool correct = checkResults(buffer, batchSize, numOps);
//			if (correct)
//				printf("PASSED!\n");
//			else
//				printf("FAILED!\n");
		}
		startIndex++;
    }

	// collect metrics before exit.
//	gpuThreadArgs->threadArgs->ProcLatencyResults[threadId] = metrics::avgProcLatency(recvTimes, procTimes, startIndex, 0);
//	gpuThreadArgs->threadArgs->throughputResults[threadId] = metrics::avgProcThroughput(recvTimes, procTimes, startIndex,0);
//	gpuThreadArgs->threadArgs->numProcessedMsgs[threadId] = startIndex;

	if (threadId == 0){
		metrics::report(gpuThreadArgs->threadArgs);
	}
	std::cout << "[Info] GPU thread exiting" << std::endl;
	zmq_msg_close(&msg);
	zmq_close (receiver);
    return NULL;
}
