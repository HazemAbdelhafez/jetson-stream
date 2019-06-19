/*
 * jettson-controller.h
 *
 *  Created on: Apr 15, 2018
 *      Author: hazem
 */


#ifndef JETTSON_CONTROLLER_H_
#define JETTSON_CONTROLLER_H_

#include <iostream>
#include <fstream>
#include <string>

#define CPU_CONFIG_DIR_PREFIX "/sys/devices/system/cpu/cpu"
#define CPU_FREQ_GOV_POSTFIX "/cpufreq/scaling_governor"
#define CPU_FREQ_SCALING_POSTFIX "/cpufreq/scaling_setspeed"

#define MAX_CPU_FREQ 1734000
#define MAX_EMC_FREQ 1600000000

#define MIN_CPU_FREQ 102000
#define MIN_EMC_FREQ 40800000

#define CORES_COUNT 4
namespace jetson{
	void init();
	void setToMax();
	void setToMin();
	void setToDefault();
	bool verifyFreq(int freq);
}
#endif /* JETTSON_CONTROLLER_H_ */
