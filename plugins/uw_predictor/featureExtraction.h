#ifndef _FEATURE_EXTRACTION_H_
#define _FEATURE_EXTRACTION_H_

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/features2d.hpp"

#include "FGObject.h"

using namespace std;
using namespace cv;


void localBinaryPatterns(InputArray src, OutputArray dst, int radius);

void getHOGFeat(Mat& src, vector<float>& dst);

vector<double> getGradHist(Mat img, Mat fg);

class FeatureExtraction
{
public:

	vector<int> getCurvatureMaxIdx(const vector<Point>& contour);
	vector<double> getCurvature(const vector<Point>& contour);

	// curvature scale space
	vector<int> getConcaveIndices(const FGObject& obj);
	vector<double> getCurvature(const FGObject& obj);
	vector<pair<double, int>> getCSSMaxima(const FGObject& obj, OutputArray cssImg);
	float	cssMatchingCost(const vector<pair<double, int>>& cssMax, const vector<pair<double, int>>& cssMaxRef);

	// Fourier descriptor
	vector<double>	getFourierDescriptor(const FGObject& obj);
	vector<double>	getFourierDescriptor(const vector<Point>& contour);

	// auto correlation
	vector<double> getCorrelation(const vector<Point>& contour, bool isTailAtRight);

	// SIFT descriptor
	//void getSIFTdescriptor(const FGObject& obj, InputArray src, OutputArray dst);

	// data members
	int num;

	void setMarkers(const cv::Mat& markerImage) {
		// Convert to image of ints
		markerImage.convertTo(waterMarkers,CV_32S);
	}

	Mat process(const cv::Mat &image) {
		// Apply watershed
		watershed(image,waterMarkers);
		return waterMarkers;
	}

	// Return result in the form of an image
	Mat getSegmentation() {
		Mat tmp;
		// all segment with label higher than 255
		// will be assigned value 255
		waterMarkers.convertTo(tmp,CV_8U);
		return tmp;
	}

	// Return watershed in the form of an image
	Mat getWatersheds() {
		Mat tmp;
		waterMarkers.convertTo(tmp,CV_8U,255,255);
		return tmp;
	}

private:
	// curvature scale space methods
	void	cssImage(vector<Point> contour, OutputArray cssImg);
	
	// Fourier descriptor methods
	vector<complex<double>>	discreteFourierTransform(const vector<double>& series);
	float principalOrientation(Mat img, int x, int y, int size);

	Mat waterMarkers;
};

class ModelLearner
{
public:
};

#endif