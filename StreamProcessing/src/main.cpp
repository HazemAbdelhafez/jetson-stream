
#include "cpuRoutines.h"


extern "C"{
	#include "zmq_draft.h"
}

int main(int argc, char **argv){

	// Check the number of input arguments
	if (argc-1 < NUM_ARGS){
		fprintf(stderr,"[Error] Invalid number of arguments. Required %d but given %d. Exiting!\n", NUM_ARGS, argc-1);
		return -1;
	}

	cxxopts::Options options("StreamServer", "Test program for stream processing.");
	// Declare and set default values for command lines
	options.add_options()
			("p,producerAddress", "The IP interface producer", cxxopts::value<std::string>()->default_value(DEFAULT_SERVER_ADDRESS))
			("c,consumerAddress", "The IP interface consumer", cxxopts::value<std::string>()->default_value(DEFAULT_SERVER_ADDRESS))
			("d,dataPort","The port on which the server receives the stream of messages", cxxopts::value<u_int>()->default_value(DEFAULT_DATA_PORT))
			("x,notifierPort","The port on which the server notifies the data-generator to start sending data", cxxopts::value<u_int>()->default_value(DEFAULT_NOTIFIER_PORT))
			("t,numThreads","Number of consumer threads to run", cxxopts::value<u_int>()->default_value(DEFAULT_NUM_WORKERS))
			("o,numOps","Number of arithmetic operations per message item", cxxopts::value<u_int>()->default_value(DEFAULT_NUM_OPS))
			("s,msgSize","The size of the message data array", cxxopts::value<u_int>()->default_value(DEFAULT_MSG_SIZE))
			("m,computeMode","Compute mode of the running test CPU:0, GPU: 1, Both: 2", cxxopts::value<u_int>()->default_value(DEFAULT_COMPUTE_MODE))
			("b,batchSize","Batch size for GPU kernel launches", cxxopts::value<u_int>()->default_value(DEFAULT_BKERNEL_BATCH_SIZE))
			("n,numMsgs","Total number of messages in this test", cxxopts::value<u_int>()->default_value(DEFAULT_NUM_MSGS))
			("i,numColdStartSamples","Number of cold start samples to ignore", cxxopts::value<u_int>()->default_value(COLD_START_SAMPLES))
			("j,testTime","Total test time acp", cxxopts::value<int>()->default_value(TEST_TIME))
			("v,verbose","Verbose", cxxopts::value<u_int>()->default_value(DEFAULT_LOGGING_MODE))
			("f,outFile","outFilePath", cxxopts::value<std::string>());

	auto cmdOpts = options.parse(argc, argv);


	// Parse command line arguments
	u_int dataPort = cmdOpts["d"].as<u_int>();
    u_int numThreads = cmdOpts["t"].as<u_int>();
    u_int computeMode = cmdOpts["m"].as<u_int>();
    std::string* consumerAddress = (std::string*) &cmdOpts["c"].as<std::string>();

    void *context = zmq_ctx_new ();
    std::atomic<u_int> threadsReady(0);
    std::atomic<u_int> threadsDone(0);
    std::atomic<u_int> collectedMetrics(0);

    // Initialize thread context
    ThreadArgs threadArgs;
    threadArgs.context = context;
    threadArgs.threadsReady = &threadsReady;
    threadArgs.threadsDone = &threadsDone;
    threadArgs.collectedMetrics = &collectedMetrics;
    threadArgs.numOps = cmdOpts["o"].as<u_int>();
    threadArgs.msgSize = cmdOpts["s"].as<u_int>();
    threadArgs.numMsgs = cmdOpts["n"].as<u_int>();
    threadArgs.throughputResults = (double*) malloc(numThreads*sizeof(double));
    threadArgs.ProcLatencyResults = (double*) malloc(numThreads*sizeof(double));
    threadArgs.totalLatencyResults = (double*) malloc(numThreads*sizeof(double));
    threadArgs.maxProcLatency = (double*) malloc(numThreads*sizeof(double));
    threadArgs.maxTotalLatency = (double*) malloc(numThreads*sizeof(double));
    threadArgs.minTotalLatency = (double*) malloc(numThreads*sizeof(double));
    threadArgs.minProcLatency = (double*) malloc(numThreads*sizeof(double));

    threadArgs.recvTimes.reserve(numThreads);
    threadArgs.procTimes.reserve(numThreads);

    for (int p =0; p < numThreads; p++){
	    threadArgs.recvTimes[p] = (timespec*) malloc(sizeof(timespec));
	    threadArgs.procTimes[p] = (timespec*) malloc(sizeof(timespec));
    }

    threadArgs.sumTotalLatency = (double*) malloc(numThreads*sizeof(double));
    threadArgs.sumProcLatency = (double*) malloc(numThreads*sizeof(double));

    threadArgs.numProcessedMsgs = (u_int*) malloc(numThreads*sizeof(u_int));

    threadArgs.coldStartIgnoredSamples = cmdOpts["i"].as<u_int>();
    threadArgs.testTime = (long int)  cmdOpts["j"].as<int>();
    threadArgs.outputFilePath = (std::string*) &cmdOpts["f"].as<std::string>();
    threadArgs.verbose = cmdOpts["v"].as<u_int>();
    threadArgs.numThreads = numThreads;
    threadArgs.numBLASThreads = numThreads;
    threadArgs.computeMode = computeMode ;
    threadArgs.gpuBatchSize = cmdOpts["b"].as<u_int>();

    threadArgs.singleThreaded = false;
    threadArgs.controlJetson = false;

    if (threadArgs.verbose)
    	std::cout << "[Info] Number of threads is " << numThreads << std::endl;

    // Initialize GPU thread context
    GPUThreadArgs gpuThreadArgs;
    gpuThreadArgs.threadArgs = &threadArgs;
    gpuThreadArgs.batchSize = cmdOpts["b"].as<u_int>();
    gpuThreadArgs.threadsPerBlock = DEFAULT_BKERNEL_THREADS_PER_BLOCK;

    // Initialize the notifier thread context
    NotifierArgs notifierArgs;
    notifierArgs.threadArgs = &threadArgs;
    notifierArgs.producerAddress = (std::string*) &cmdOpts["p"].as<std::string>();

    notifierArgs.notifierPort = cmdOpts["x"].as<u_int>();

    // Check that the input arguments are correct.
    if (*(threadArgs.outputFilePath) == ""){
    	std::cerr << "[Error] Output file path is not specified" << std::endl;
    	return -1;
    }

    // Socket to talk to clients
    void *clients = zmq_socket (context, ZMQ_DISH);
    int rc;
    u_int maxOutstandingMsgs = MAX_PENDING_MSGS;
    zmq_setsockopt(clients, ZMQ_RCVHWM, (void*) &maxOutstandingMsgs, sizeof(maxOutstandingMsgs));
    zmq_setsockopt(clients, ZMQ_SNDHWM, (void*) &maxOutstandingMsgs, sizeof(maxOutstandingMsgs));

    rc = zmq_join (clients, MSG_GROUP);
    assert(rc == 0);

    rc = zmq_bind (clients, getURL(UDP_PROTOC, *(consumerAddress), dataPort).c_str());
    if (rc == -1){
    	fprintf(stderr, "[Error] Couldn't bind to specified address. Error code %d\n", errno);
    	exit(-1);
    }

    // Socket to talk to workers
    void *workers = zmq_socket (context, ZMQ_PUSH);
    rc = zmq_bind (workers, WORKERS_ENDPOINT);
    assert(rc == 0);

    // Prepare thread launch parameters
	pthread_t threads[numThreads];

    // Launch pool of worker threads
    switch (computeMode){
    case CPU:
    	// Launch all threads on CPU.
        for (u_int i = 0; i < numThreads; i++) {
            pthread_create (&threads[i], NULL, workerRoutine, (void*) &notifierArgs);
        }
        break;
    case GPU:
    	// Override number of threads, in GPU case we only launch one thread to launch the kernels.
    	numThreads = 1;
    	// Launch all threads on GPU.
		for (int i = 0; i < numThreads; i++) {
			pthread_create (&threads[i], NULL, bKernelDriver, (void*) &gpuThreadArgs);
		}
        break;
    case HETEROGENEOUS:
    	break;
    case RACE_TO_FINISH:
	    // Launch ONE thread on CPU.
	    threadArgs.singleThreaded = true;
	    threadArgs.controlJetson = true;
	    pthread_create (&threads[0], NULL, workerRoutine, (void*) &notifierArgs);
	    break;
    case PER_MESSAGE_PARALLELISM:
	    threadArgs.singleThreaded = true;
	    // Launch ONE thread on CPU.
	    pthread_create (&threads[0], NULL, workerRoutine, (void*) &notifierArgs);
	    break;
    default:
    	printf("Invalid GPU kernel mode.\n");
    	break;
    }
    // Create a thread that notifies the client that we are ready.
    pthread_t notifier_t;
    pthread_create(&notifier_t, NULL, startProducer, (void*) &notifierArgs);

    //  Connect work threads to client threads via a queue proxy
    rc = zmq_proxy (clients, workers, NULL);

    if (rc!=0){
    	// Interrupt signal might get us here.
    	if (threadArgs.verbose)
    		printf("[Shutdown] Server interrupted, shutting down gracefully.\n");
		if (computeMode == GPU){
				if (threadArgs.verbose)
					printf("[Shutdown] Reset the GPU.\n");
				// Clear GPU resources
				cudaDeviceReset();
		}
		// Free allocated buffers
		free(threadArgs.throughputResults);
		free(threadArgs.ProcLatencyResults);
		free(threadArgs.numProcessedMsgs);
		free(threadArgs.totalLatencyResults);
		free(threadArgs.sumProcLatency);
		free(threadArgs.sumTotalLatency);

    }
    return 0;
}
