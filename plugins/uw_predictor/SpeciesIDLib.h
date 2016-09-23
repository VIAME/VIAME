//////////////////////////////////////////////////////////////////////////
//
//  UWEESpeciesIDLib.h
//  Date:   Jun/12/2014
//  Author: Meng-Che Chuang, University of Washington
//
//  Header file for fish species classifier
//

#ifndef _UWEE_SPECIES_ID_LIB_H_
#define _UWEE_SPECIES_ID_LIB_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "classHierarchy.h"
#include "featureExtraction.h"
#include "FGExtraction.h"
#include "FGObject.h"

using namespace std;
using namespace cv;


// Hierarchical Partial Classifier for fish species
//   - a tree of binary _svms
//   - partial classification
//   - gives probabilistic output
class FishSpeciesID
{
public:
	FishSpeciesID();
	~FishSpeciesID();

	// load classifier model from a file
	void loadModel(const char* filename);

	// save classifier model to a file
	void saveModel(const char* filename);

	// output features
	int outputFeature(Mat img, Mat img2, Mat& feature);

	// training routine
	void train(Mat data, Mat labels);

	// testing routine
	bool predict(Mat img, Mat img2, vector<int>& predictions, vector<double>& probabilities, Mat &fgRect);

	int getDimFeat() {return _dimFeat;};

private:
	// feature extraction from the training/testing image
	int extractFeatures(Mat src, Mat src2, Mat& features, bool useTailFD, bool useHeadFD, Mat &fgRect, Point& shift, Mat& rotateR);

	ClassHierarchy _classHierarchy;
	static int _dimFeat;
	int _count;

public:
	//////////////////////////////////////////////////////////////////////////
	// segmentation parameters

	// threshold for filtering the background net patterns
	int		_thresh;

	// structuring element size for filtering the background net patterns
	int		_seLength;

	// minimum object area to be preserved
	double	_minArea;

	// maximum object area to be preserved
	double	_maxArea;

	// minimum object aspect ratio (i.e. max(w/h, h/w)) to be preserved
	double	_minAspRatio;

	// maximum object aspect ratio (i.e. max(w/h, h/w)) to be preserved
	double	_maxAspRatio;

	// threshold for ratio histogram backprojection
	double	_histBP_theta;
};

#endif
