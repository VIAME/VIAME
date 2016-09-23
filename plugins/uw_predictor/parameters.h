#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

#include <iostream>
#include <vector>
using namespace std;

// preprocessors
#define DEBUG
//#define GENERATE_FEATURES

// enum: fish species
enum FishSpecies {
	SPC_KING_SALMON = 0,
	SPC_CHUM_SALMON,
	SPC_JUVENILE_POLLOCK,
	SPC_ADULT_POLLOCK,
	SPC_CAPELIN,
	SPC_EULACHON,
	SPC_ROCKFISH,
	// dummy
	SPC_DUMMY
};

// enum: fish groups
enum FishGroups {
	GROUP_SALMON = 0,
	GROUP_GADOID,
	GROUP_SMELT,
	GROUP_ROCKFISH,
	// dummy
	GROUP_DUMMY
};

// class: Parameters
class Parameters
{
public:
	Parameters();
	~Parameters();

	int getFrameWidth() const { return frameWidth; }
	int getFrameHeight() const { return frameHeight; }

	int getOutWidth() const { return outWidth; }
	int getOutHeight() const { return outHeight; }

	char* getInputPath() const { return (char*)inputPath; }
	char* getInputPath2() const { return (char*)inputPath2; }
	char* getResultPath() const { return (char*)resultPath; }

	int getNumInstances() const { return numInstances; }
	int getDimFeatures() const { return dimFeatures; }
	int getNumClasses() const { return numClasses; }
	char* getClassLabel(const size_t& i);
	char* getGroupLabel(const size_t& i);



	void setResultPath(const char* path) { snprintf(resultPath, 100, "%s", path); }
	void setDimFeatures(const int& k) { dimFeatures = k; }
	void setNumInstances(const int& n) { numInstances = n; }
	void setNumClasses(const int& c) { numClasses = c; }


private:
	// frame width
	int frameWidth;

	// frame height
	int frameHeight;


	// output frame width
	int outWidth;

	// output frame height
	int outHeight;


	// path of input
	char inputPath [256];
	char inputPath2 [256];

	// path of output
	char resultPath [256];


	// number of instances
	int numInstances;

	// dimension of feature vectors
	int dimFeatures;

	// total number of classes
	int numClasses;

	// class names
	vector<char*> classLabels;

	// group names
	vector<char*> groupLabels;


	// minimum aspect ratio
	double minAspRatio;

	// maximum aspect ratio
	double maxAspRatio;


};

#endif
