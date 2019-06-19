/*
 * config.h
 *
 *  Created on: Dec 18, 2017
 *      Author: hazem
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#define MSG_GROUP "d"

#define UDP_PROTOC "udp"
#define TCP_PROTOC "tcp"
#define INPROC_PROTOC "inproc"

#define MEGA 1000
#define GIGA (unsigned int) 1000*MEGA

#define DEFAULT_SERVER_ADDRESS "127.0.0.1"
#define DEFAULT_DATA_PORT "5555"
#define DEFAULT_NOTIFIER_PORT "5554"
#define DEFAULT_NUM_WORKERS "1"
#define DEFAULT_MSG_SIZE "100"
#define DEFAULT_NUM_OPS "1000"
#define DEFAULT_COMPUTE_MODE "10"
#define DEFAULT_BKERNEL_BATCH_SIZE "640"
#define DEFAULT_BKERNEL_THREADS_PER_BLOCK 128
#define DEFAULT_NUM_MSGS "40000"
#define DEFAULT_LOGGING_MODE "1"
#define MAX_MSG_SIZE 4096
#define WORKERS_ENDPOINT "inproc://workers"
#define MAX_PENDING_MSGS 200
#define MSG_STATISTICS_BUFFER_SIZE 50000

#define SAXPY 1
#define DOT 2
#define EUCLIDEAN_DIST 4
#define MAT_VEC_MUL 64
#define MAT_MAT_MUL 8

#define COLD_START_SAMPLES "10"
#define TEST_TIME "30"
#define NUM_ARGS 14

enum ComputeMode{
	CPU,
	GPU,
	HETEROGENEOUS,
	RACE_TO_FINISH,
	PER_MESSAGE_PARALLELISM
};

enum GPU_MODE{
	PERSISTENT,
	BATCH
};

#define GPU_MODE_DEFAULT BATCH
#endif /* CONFIG_H_ */
