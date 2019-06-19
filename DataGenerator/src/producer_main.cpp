/*
 * main.c
 *
 *  Created on: Dec 13, 2017
 *      Author: hazem
 */


#include <pthread.h>
#include <unistd.h>
#include "zmq.h"
#include <stdlib.h>
#include <string.h>
#include "assert.h"
#include <iostream>
#include "timers.h"
#include "cxxopts.hpp"
#include "fstream"
#include <math.h>
#include <iomanip>
#include <atomic>
extern "C"{
#include "zmq_draft.h"
}

typedef double item_t;

#define MSG_GROUP "d"
#define NUM_ARGS 10
#define DEFAULT_NUM_MSGS "100"
#define DEFAULT_MSG_SIZE "100"
#define UDP_PROTOC "udp"
#define TCP_PROTOC "tcp"
#define DEFAULT_SERVER_ADDRESS "127.0.0.1"
#define DEFAULT_DATA_PORT "5555"
#define DEFAULT_NOTIFIER_PORT "5555"
#define DEFAULT_CALL_COUNT "10"
#define DEFAULT_THROUGHPUT "100"
#define DEFAULT_LOGGING_MODE "0"
#define COLD_START_SAMPLES "10"
#define MAX_BYTES_THROUGHPUT 64*1024*1024 // 128 MB is the maximum bandwidth allowed on the Jetson, we take half as a practical limit
#define MAX_MSG_SIZE 4096
#define OUTPUT_DELIMETER " "
struct ThreadContext{
	void* context;
	std::string* producerAddress;
	std::string* consumerAddress;
	u_int dataPort;
	u_int notifierPort;
	u_int numMsgs;
	u_int msgSize;
	u_int throughput;
	u_int delayCallCount;
	u_int coldStartIgnoredSamples;
	u_int verbose;
	volatile u_int consumerDone;
	std::string* resultsDirPath;
	std::string* outputFilePath;
	std::atomic<u_int>* consumerReady;
};


void delete_buffer(void *buffer, void *hint){
	free(buffer);
}

template <class T>
void initData(T* data, unsigned int n){
	int initValue = 10;
	for(unsigned int i=0; i<n; i++)
		data[i] = (T) initValue*i;
}

template <class T>
void printData(T* data, unsigned int n){
	std::cout << "[Info] Sent data: [";
	std::cout << data[0];
	if (n==1)
		return;
	for(unsigned int i=1; i<n;i++){
		std::cout << ", " << data[i];
	}
	std::cout << "]" <<std::endl;

}

std::string getURL(std::string proto, std::string address, u_int port){
	std::string url = proto;
	url.append("://");
	url.append(address);
	url.append(":");
	url.append(std::to_string(port));
	return url;
}

void* startProducer(void* cxt){
	void *notifier = zmq_socket (((ThreadContext*) cxt)->context, ZMQ_PAIR);
	int rc = zmq_bind(notifier, getURL(TCP_PROTOC, *(((ThreadContext*) cxt)->producerAddress), ((ThreadContext*) cxt)->notifierPort).c_str());
	if(rc == -1){
		std::cerr << "[Error] Failed to bind to the specified address for the producer start notifier" << std::endl;
		std::cerr << "Error code is: " << errno << std::endl;
	}

	assert(rc==0);
	while(1){
		char msg[5];
		rc = zmq_recv(notifier, msg, 5, 0);
		if (rc == -1)
			break;
		assert(rc == 5);
		if (((ThreadContext*) cxt)->verbose)
			printf("[Info] Consumer is ready.\n");
		break;
	}
	ThreadContext* threadContext = (ThreadContext*) cxt;
	std::atomic<u_int>* ready = threadContext->consumerReady;
	*ready += 1;
	zmq_close(notifier);
	return NULL;
}

void* stopProducer(void* cxt){
	void *notifier = zmq_socket (((ThreadContext*) cxt)->context, ZMQ_REP);
	int rc = zmq_bind(notifier, getURL(TCP_PROTOC, *(((ThreadContext*) cxt)->producerAddress), ((ThreadContext*) cxt)->notifierPort + 5).c_str());
	if(rc == -1){
		std::cerr << "[Error] Failed to bind to the specified address for the producer stop notifier" << std::endl;
		std::cerr << "Error code is: " << errno << std::endl;
	}
	while(1){
		char msg[5];
		rc = zmq_recv(notifier, msg, 5, 0);
		if (rc == -1)
			break;
		assert(rc == 5);
		((ThreadContext*) cxt)->consumerDone = 1;
		zmq_send(notifier, "done", 4, 0);
		if (((ThreadContext*) cxt)->verbose)
			printf("[Info] Consumer is done.\n");
		break;
	}
	zmq_close(notifier);
	return NULL;
}


double nextTimeDiff(double throughput)
{
    return -log(1.0 - (double) rand() / (RAND_MAX + 1.0)) / throughput;
}


void* exponential_generator(void* tcxt){
	ThreadContext* cxt = (ThreadContext*) tcxt;
	u_int n = cxt->msgSize;
	u_int numMsgs = cxt->numMsgs;
	u_int throughput = cxt->throughput; // Throughput : messages / second
	size_t msgSize = n*sizeof(item_t);
	char *group = (char*) MSG_GROUP;

	//  A UDP socket to send data samples to the server.
	void *data_generator = zmq_socket (cxt->context, ZMQ_RADIO);
	int maxOutstandingMsgs = numMsgs;						// Maximum number of outgoing messages to queue
	int sendKernelBuff = maxOutstandingMsgs*sizeof(item_t)*msgSize;
	zmq_setsockopt(data_generator, ZMQ_SNDBUF , (void*) &sendKernelBuff, sizeof(sendKernelBuff));
	zmq_setsockopt(data_generator, ZMQ_SNDHWM, (void*) &maxOutstandingMsgs, sizeof(maxOutstandingMsgs));

	zmq_connect (data_generator, getURL(UDP_PROTOC, *(cxt->consumerAddress), cxt->dataPort).c_str());
	item_t* data = (item_t*) malloc(msgSize);

	initData(data, n);

	void *hint = NULL;

	// Initialize a msg with zero-copying.
	zmq_msg_t refMsg;
	zmq_msg_init_data (&refMsg, data, msgSize, delete_buffer, hint);
	zmq_msg_set_group(&refMsg, group);

	zmq_msg_t msg;
	zmq_msg_init_size(&msg, msgSize);
	zmq_msg_copy(&msg, &refMsg);

	timespec tp1, tp2;
	int rc = 0;
	u_int sentMsgs = 0;


	double timeDiff = 0;
	timespec start, current, trigger;
	timeDiff = nextTimeDiff(throughput);
	// Wait until I tell the consumer that the producer is ready.
	std::atomic<u_int>* consumerReady = cxt->consumerReady;
	while (*consumerReady != 1){
		usleep(1000);
		continue;
	}
	timer::markTime(&start);
	timer::markTime(&trigger);
	while(!cxt->consumerDone){
		// Mark the start time after ignoring cold start samples
		timer::markTime(&current);
		if (timer::getDuration(current, trigger) > 0)
			continue;
		if (sentMsgs == cxt->coldStartIgnoredSamples-1) // Since we ignore N-1 from front and 1 from end in the consumer{
			timer::markStart(&tp1);
		rc = zmq_msg_send(&msg, data_generator, 0);
		if ((u_int) rc != msgSize && cxt->verbose)
			fprintf(stderr, "[Error] Error while sending message. Error code is %d\n", errno);
		else
			sentMsgs++;
		zmq_msg_copy(&msg, &refMsg);
		timeDiff = nextTimeDiff(throughput);
		long int  tv_sec = floor(timeDiff);
		long int tv_nsec = (timeDiff  -  tv_sec)*BILLION;
		trigger = timer::addTime(trigger, tv_sec, tv_nsec);
	}
	timer::markStop(&tp2);
	double duration = timer::getDuration(tp1, tp2);
	double achievedThroughput = (sentMsgs-cxt->coldStartIgnoredSamples+1)/duration;

	// Start time and end time stamps of the data generator, I use them to synchronize the power readings.
	double startTimeStamp = tp1.tv_sec + (double) tp1.tv_nsec/BILLION;
	double stopTimeStamp = tp2.tv_sec + (double) tp2.tv_nsec/BILLION;
	if (cxt->verbose){
		if (sentMsgs != numMsgs){
			std::cout << "[Warning] Number of sent messages is not equal to the number of required messages." << std::endl;
		}
		std::cout<< "[Info] Sent " << sentMsgs << " messages in " << duration << " seconds"<< std::endl;
		std::cout<< "[Info] Average achieved sending throughput: " << achievedThroughput<< " messages/second"<<std::endl;
		std::cout << "[Info] Duration " << duration << std::endl;
	}
	// Report the performance numbers
	std::ofstream outputFile;
	outputFile.open(*(cxt->outputFilePath));
	outputFile << numMsgs << OUTPUT_DELIMETER
			<< cxt->throughput << OUTPUT_DELIMETER
			<< sentMsgs << OUTPUT_DELIMETER
			<< achievedThroughput<<OUTPUT_DELIMETER
			<< std::setprecision(15) << startTimeStamp<<OUTPUT_DELIMETER
			<< std::setprecision(15) << stopTimeStamp << std::endl;
	outputFile.close();
	// Clean up
	zmq_msg_close(&msg);
	zmq_msg_close(&refMsg);
	zmq_close(data_generator);
	return NULL;
}

//TODO: TIMING IS NOT ACCURATE. UPDATE FROM THE EXPONENTIAL GENERATOR
void* generator(void* tcxt){
	ThreadContext* cxt = (ThreadContext*) tcxt;
	u_int n = cxt->msgSize;
	u_int numMsgs = cxt->numMsgs;
	u_int throughput = cxt->throughput; // Throughput : messages / second
	size_t msgSize = n*sizeof(item_t);
	char *group = (char*) MSG_GROUP;

	double msgDelay = (double) 1/throughput;
	u_int sendBatchSize = numMsgs/cxt->delayCallCount; 		// Make sure we call nanosleep only 10 times to reduce error margin between the achieved throughput and the required one.

	double  overAllSleepTime = msgDelay * sendBatchSize;
	long   sleepSeconds  = (long) static_cast<u_int>(floor(overAllSleepTime));
	struct timespec delay;
	if (sleepSeconds <= 0)
		delay = {0, (long) static_cast<u_int>(overAllSleepTime*BILLION)};
	else
		delay = { (long) static_cast<u_int>(sleepSeconds), (long) static_cast<u_int>((overAllSleepTime - sleepSeconds)*BILLION)};
	if (cxt->verbose){
		std::cout << "[Info] Delay value is " <<delay.tv_sec << " seconds and "<< delay.tv_nsec << " ns" <<std::endl;
	}
	//  A UDP socket to send data samples to the server.
	void *data_generator = zmq_socket (cxt->context, ZMQ_RADIO);
	int maxOutstandingMsgs = numMsgs;						// Maximum number of outgoing messages to queue
	int sendKernelBuff = maxOutstandingMsgs*sizeof(item_t)*msgSize;
	zmq_setsockopt(data_generator, ZMQ_SNDBUF , (void*) &sendKernelBuff, sizeof(sendKernelBuff));
	zmq_setsockopt(data_generator, ZMQ_SNDHWM, (void*) &maxOutstandingMsgs, sizeof(maxOutstandingMsgs));

	zmq_connect (data_generator, getURL(UDP_PROTOC, *(cxt->consumerAddress), cxt->dataPort).c_str());
	item_t* data = (item_t*) malloc(msgSize);

	initData(data, n);

	void *hint = NULL;

	// Initialize a msg with zero-copying.
	zmq_msg_t refMsg;
	zmq_msg_init_data (&refMsg, data, msgSize, delete_buffer, hint);
	zmq_msg_set_group(&refMsg, group);

	zmq_msg_t msg;
	zmq_msg_init_size(&msg, msgSize);
	zmq_msg_copy(&msg, &refMsg);

	timespec tp1, tp2;
	int rc = 0;
	u_int sentMsgs = 0;

	while(!cxt->consumerDone){
		for (u_int i=0; i<sendBatchSize; i++){
			// Mark the start time after ignoring cold start samples
			if (sentMsgs == cxt->coldStartIgnoredSamples-1) // Since we ignore N-1 from front and 1 from end in the consumer
				timer::markStart(&tp1);
			rc = zmq_msg_send(&msg, data_generator, 0);
			if ((u_int) rc != msgSize && cxt->verbose)
				fprintf(stderr, "[Error] Error while sending message. Error code is %d\n", errno);
			else
				sentMsgs++;
			zmq_msg_copy(&msg, &refMsg);
		}
		nanosleep(&delay, NULL);
	}
	timer::markStop(&tp2);
	double duration = timer::getDuration(tp1, tp2);
	double achievedThroughput = (sentMsgs-cxt->coldStartIgnoredSamples+1)/duration;

	// Start time and end time stamps of the data generator, I use them to synchronize the power readings.
	double startTimeStamp = tp1.tv_sec + (double) tp1.tv_nsec/BILLION;
	double stopTimeStamp = tp2.tv_sec + (double) tp2.tv_nsec/BILLION;
	if (cxt->verbose){
		if (sentMsgs != numMsgs){
			std::cout << "[Warning] Number of sent messages is not equal to the number of required messages." << std::endl;
		}
		std::cout<< "[Info] Sent " << sentMsgs << " messages in " << duration << " seconds"<< std::endl;
		std::cout<< "[Info] Average achieved sending throughput: " << achievedThroughput<< " messages/second"<<std::endl;
	}
	// Report the performance numbers
	std::ofstream outputFile;
	outputFile.open(*(cxt->outputFilePath));
	outputFile << numMsgs << OUTPUT_DELIMETER
			<< cxt->throughput << OUTPUT_DELIMETER
			<< sentMsgs << OUTPUT_DELIMETER
			<< achievedThroughput<<OUTPUT_DELIMETER
			<< std::setprecision(15) << startTimeStamp<<OUTPUT_DELIMETER
			<< std::setprecision(15) << stopTimeStamp << std::endl;
	outputFile.close();
	// Clean up
	zmq_msg_close(&msg);
	zmq_msg_close(&refMsg);
	zmq_close(data_generator);
	return NULL;
}

int main(int argc, char **argv) {
	// Check the number of input arguments
	if (argc-1 < NUM_ARGS){
		fprintf(stderr,"[Error] Invalid number of arguments. Required %d but given %d. Exiting!\n", NUM_ARGS, argc-1);
		return -1;
	}

	cxxopts::Options options("DataGenerator", "Test program for stream-data generation.");
	// Declare and set default values for command lines
	options.add_options()
					("p,producerAddress", "The IP interface producer", cxxopts::value<std::string>()->default_value(DEFAULT_SERVER_ADDRESS))
					("c,consumerAddress", "The IP interface consumer", cxxopts::value<std::string>()->default_value(DEFAULT_SERVER_ADDRESS))
					("d,dataPort","The port on which the server receives the stream of messages", cxxopts::value<u_int>()->default_value(DEFAULT_DATA_PORT))
					("x,notifierPort","The port on which the server notifies the data-generator to start sending data", cxxopts::value<u_int>()->default_value(DEFAULT_NOTIFIER_PORT))
					("n,numMsgs","Number of messages to send per program run", cxxopts::value<u_int>()->default_value(DEFAULT_NUM_MSGS))
					("s,msgSize","The size of the message data array", cxxopts::value<u_int>()->default_value(DEFAULT_MSG_SIZE))
					("t,throughput","Throughput as messages per second", cxxopts::value<u_int>()->default_value(DEFAULT_THROUGHPUT))
					("m,delayCallCount","The number of times to call the sleep function to achieve a specific throughput", cxxopts::value<u_int>()->default_value(DEFAULT_CALL_COUNT))
					("i,numColdStartSamples","Number of cold start samples to ignore", cxxopts::value<u_int>()->default_value(COLD_START_SAMPLES))
					("v,verbose","Verbose", cxxopts::value<u_int>()->default_value(DEFAULT_LOGGING_MODE))
					("f,outFile","outFilePath", cxxopts::value<std::string>());
	auto cmdOpts = options.parse(argc, argv);


	void *context = zmq_ctx_new ();
	zmq_ctx_set(context, ZMQ_IO_THREADS, 2);
	ThreadContext cxt;
	cxt.context = context;
	cxt.producerAddress = (std::string*) &cmdOpts["p"].as<std::string>();
	cxt.consumerAddress = (std::string*) &cmdOpts["c"].as<std::string>();
	cxt.numMsgs = cmdOpts["n"].as<u_int>();
	cxt.msgSize = cmdOpts["s"].as<u_int>();
	cxt.dataPort = cmdOpts["d"].as<u_int>();
	cxt.notifierPort = cmdOpts["x"].as<u_int>();
	cxt.throughput = cmdOpts["t"].as<u_int>();
	cxt.delayCallCount = cmdOpts["m"].as<u_int>();
	cxt.coldStartIgnoredSamples = cmdOpts["i"].as<u_int>();
	cxt.verbose = cmdOpts["v"].as<u_int>();
	cxt.outputFilePath = (std::string*) &cmdOpts["f"].as<std::string>();
	cxt.consumerDone = 0;

	std::atomic<u_int> consumerReady(0);
	cxt.consumerReady = &consumerReady;

	if (cxt.msgSize > MAX_MSG_SIZE){
		// TODO: De-bug this issue, when we use msg with more than 4096 items, the zmq_close
		// returns a sigabort due to freeing an already freed address by the IO thread or
		// the reaper thread.
		fprintf(stderr, "[Error] Cannot send messages that have more than 4k items.");
		return -1;
	}

	if (cxt.throughput * cxt.msgSize > MAX_BYTES_THROUGHPUT){
		std::cerr << "[Error] Throughput cannot exceed the maximum bandwidth on the Jetson board." <<std::endl;
		std::cerr << "[More Details: " << std::endl << "  - Required throughput is " << cxt.throughput*cxt.msgSize/(1024*1024) << " MB/Sec"
				<< std::endl << "  - Maximum allowed throughput is " << MAX_BYTES_THROUGHPUT << " MB/sec" << std::endl;
		return -1;
	}

	if (*(cxt.outputFilePath) == ""){
		std::cerr << "[Error] Output file path is not specified" << std::endl;
		return -1;
	}
	if (cxt.verbose){
		printf("[Info] Server address is: %s:%d\n", (*cxt.consumerAddress).c_str(), cxt.dataPort);
		printf("[Info] Notifier address is: %s:%d\n", (*cxt.producerAddress).c_str(), cxt.notifierPort);
		printf("[Info] Number of messages: %u\n", cxt.numMsgs);
		printf("[Info] Message size: %u\n", cxt.msgSize);
		printf("[Info] Required throughput: %u msgs/sec \n", cxt.throughput);
		printf("[Info] Sleep function calls count: %u\n", cxt.delayCallCount);
	}
	// Create a thread that waits for a notification that the server is ready
	// before launching the data generator.
	pthread_t notifier_t;
	pthread_create (&notifier_t, NULL, startProducer, (void*) &cxt);
//	pthread_join(notifier_t, NULL); // Wait for a notification from the server to start.

	pthread_t generator_t;
	pthread_create (&generator_t, NULL, exponential_generator, (void*) &cxt);

	pthread_t stopNotifier;
	pthread_create(&stopNotifier, NULL, stopProducer, (void*) &cxt);
	pthread_join(stopNotifier, NULL);
	zmq_ctx_destroy (context);


	return 0;
}
