#ifndef _FGEXTRACTION_OPEN_H_
#define _FGEXTRACTION_OPEN_H_

#include <string>
#include <vector>
#include <cmath>
#include <assert.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "FGObject.h"
//#include "parameters.h"
#include "util.h"

using namespace std;
using namespace cv;

class FGExtraction
{
public:
	FGExtraction() {}
	~FGExtraction() {}

	// getters
	Mat getBgsImg() const { return _bgsImg; }
	Mat getFgImg() const { return _fgImg; }

	// setters

	// main method - foreground extraction
	vector<FGObject>* extractFGTargets(InputArray inImg, OutputArray fgImg, int seLength, int threshVal, 
		                               double minArea, double maxArea, 
		                               double minAspRatio, double maxAspRatio);

private:
	// initializer
	void init();

    // background subtraction methods
    int subtractBGOpenDiagonal(InputArray src, OutputArray dst, int threshVal, int seLength);
	void subtractBGMedian(InputArray src, OutputArray dst, int threshVal, int seLength);
    
    // threshold by value methods
	void doubleThresholdByValue(double percentage, Mat roiMask);
	int getOtsuThreshold(int lowerVal, int upperVal, int* u1Ptr, InputArray roiMask);
    
    // histogram methods
    void updateByHistBackproject(double theta, Mat roiMask);
	void myUpdateByHistBackproject(double theta, Mat roiMask);
    void calcHist(Mat fg, vector<int>& hist, int histSize, Mat roiMask);
    
	Mat histBackProject(InputArray hist, Mat roiMask);
	
	void thresholdByAreaRatioVar(double minArea, double maxArea, int dilateSESize, int erodeSESize, 
								 double minAspRatio, double maxAspRation, int varThresh = 30);
	
	void postProcessing(InputArray src, OutputArray dst);
	void medianFilter(Mat src, Mat dst, int ksize, bool isHorizontal);

	// data members
	Mat		_inImg;
	Mat		_bgsImg;
	Mat		_fgHighImg;
	Mat		_fgLowImg;
	Mat		_highMask;
	Mat		_lowMask;
	Mat		_fgImg;
	Mat		_ratioHistBP;

};

#endif
