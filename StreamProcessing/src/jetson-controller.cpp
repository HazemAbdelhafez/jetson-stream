
#include "jetson-controller.h"
#include <string.h>

void writeToFile(std::string fileName, std::string value){
	std::ofstream outputFile;
	outputFile.open(fileName);
	outputFile << value;
	outputFile.close();
}

void writeToFile(std::string fileName, int value){
	std::ofstream outputFile;
	outputFile.open(fileName);
	outputFile <<  std::to_string(value);
	outputFile.close();
}

void setCPUFreq(int cpuID, int freq, bool setGoverner){
	if (setGoverner){
		std::string governerFile = CPU_CONFIG_DIR_PREFIX + std::to_string(cpuID) + CPU_FREQ_GOV_POSTFIX;
		writeToFile(governerFile, "userspace");
	}
	std::string scalingFile = CPU_CONFIG_DIR_PREFIX + std::to_string(cpuID) + CPU_FREQ_SCALING_POSTFIX;
	writeToFile(scalingFile, freq);
}

void setEMCFreq(int freq, bool override){

	writeToFile("/sys/kernel/debug/clock/override.emc/rate",  freq);
	if (override){
		std::string overrideFile = "/sys/kernel/debug/clock/override.emc/state";
		writeToFile(overrideFile, 1);
	}
}
namespace jetson{
void init(){
	std::string governerFile = "";
	for (int cpuID=0; cpuID < CORES_COUNT; cpuID++){
		governerFile = CPU_CONFIG_DIR_PREFIX + std::to_string(cpuID) + CPU_FREQ_GOV_POSTFIX;
		writeToFile(governerFile, "userspace");
	}
}
void setToMax(){
	for (int cpuID=0; cpuID < CORES_COUNT; cpuID++){
		setCPUFreq(cpuID, MAX_CPU_FREQ, false);
	}
	setEMCFreq(MAX_EMC_FREQ, true);
}
void setToMin(){
	for (int cpuID=0; cpuID < CORES_COUNT; cpuID++){
		setCPUFreq(cpuID, MIN_CPU_FREQ, false);
	}
	setEMCFreq(MIN_EMC_FREQ, true);
}

void setToDefault(){
	std::string governerFile = "";
	for (int cpuID=0; cpuID < CORES_COUNT; cpuID++){
		governerFile = CPU_CONFIG_DIR_PREFIX + std::to_string(cpuID) + CPU_FREQ_GOV_POSTFIX;
		writeToFile(governerFile, "interactive");
	}
	// Disable EMC clock override
	setEMCFreq(MIN_EMC_FREQ, false);
	writeToFile("/sys/kernel/debug/clock/override.emc/state", 0);
}

bool verifyFreq(int freq){
	bool isSet = false;
	std::string governerFile = "";
	for (int cpuID=0; cpuID < CORES_COUNT; cpuID++){
			governerFile = CPU_CONFIG_DIR_PREFIX + std::to_string(cpuID) + CPU_FREQ_SCALING_POSTFIX;
			std::string line;
			std::ifstream myfile (governerFile);
			if (myfile.is_open())
			{
				while ( getline (myfile,line) )
				{
					if (std::to_string(freq) == line){
						std::cout<< "Required frequency: " << std::to_string(freq) << " Found " << line <<std::endl;
						return true;
					} else
						std::cout<< "Required frequency: " << std::to_string(freq) << " Found " << line <<std::endl;
				}
				myfile.close();
			} else{
				std::cout << "Unable to open file" << std::endl;
				std::cout << governerFile << std::endl;
				std::cerr<< strerror(errno) <<std::endl;
			}
		}
	return isSet;
}
}
