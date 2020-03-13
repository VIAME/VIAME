#include "parameters.h"

Parameters::Parameters()
{
	// frame width (pixels)
	frameWidth = 2048;

	// frame height (pixels)
	frameHeight = frameWidth;


	// output frame width (pixels)
	outWidth = 1024;

	// output frame height (pixels)
	outHeight = outWidth;


	// path of input
	//snprintf(inputPath, 200, "D:\\species_classification\\dataset");
	snprintf(inputPath, 200, "C:\\Users\\ipl333\\Documents\\Code\\CamTrawl\\Image Dataset\\");
	snprintf(inputPath2, 200, "C:\\Users\\ipl333\\Documents\\Code\\CamTrawl\\Image Dataset\\img2\\");
	//snprintf(inputPath, 200, "D:\\Birds-200");

	// path of output
	//snprintf(resultPath, 200, "D:\\species_classification\\dataset\\results");
	snprintf(resultPath, 200, "C:\\Users\\ipl333\\Documents\\Code\\CamTrawl\\Image Dataset\\result\\");
	//snprintf(resultPath, 200, "D:\\Birds-200\\results");


	// number of classes
	numClasses = 5;

	// number of instances
	numInstances = 1026;//1325;

	// dimension of feature vectors
	dimFeatures = 75;

	// class names
	classLabels.push_back("Eulachon");
	classLabels.push_back("Pollock");
	classLabels.push_back("Rockfish");
	classLabels.push_back("Salmon");
	classLabels.push_back("Squid");
	//classLabels.push_back("Eulachon");
	//classLabels.push_back("Rockfish");
	//classes.push_back("Jellyfish_Unident");
	//classes.push_back("Shrimp_Unident");
	//classes.push_back("Squid_Unident");

	groupLabels.push_back("Salmon");
	//groupLabels.push_back("Salmon");
	groupLabels.push_back("Gadoid");
	//groupLabels.push_back("Gadoid");
	groupLabels.push_back("Smelt");
	//groupLabels.push_back("Smelt");
	groupLabels.push_back("Rockfish");


	// minimum aspect ratio
	minAspRatio = 1.8;

	// maximum aspect ratio
	maxAspRatio = 8.0;

}

Parameters::~Parameters()
{
}

char* Parameters::getClassLabel( const size_t& i )
{
	if(i < 0 || i >= classLabels.size()) return 0;
	return classLabels[i];
}

char* Parameters::getGroupLabel( const size_t& i )
{
	if(i < 0 || i >= groupLabels.size()) return 0;
	return groupLabels[i];
}
