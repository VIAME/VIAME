#ifndef _UTIL_H_
#define _UTIL_H_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <sstream>

//#include "FGObject.h"
//#include "parameters.h"

using namespace cv;

extern RNG rng;
//extern Parameters param;

// math constant PI
const double PI_ = 3.141592654;

const float body_ratio[5] = {5.5/6.5, 4.8/5.8, 0.8, 5.0/6.0, 5.2/6.2};

void showImage(const string& winname, Mat img, int autosize = 0, int delay = 0);
vector<vector<Point> > extractContours(const Mat& img);
void plotOrientedBoundingBox(Mat& img, const RotatedRect& orientedBox, Scalar color = Scalar());

RotatedRect orientedBoundingBox(const vector<Point>& contour);

template<class T>
string numToStr(const T& num)
{
	std::ostringstream ss;
	ss << num;
	return ss.str();
}

Mat descriptorHOG(const Mat& gradX, const Mat& gradY);
Mat HOG(const Mat& gradX, const Mat& gradY);
void nonMaxSuppression(const Mat& src, const int sz, Mat& dst, const Mat mask);

vector<vector<int>> combnk(vector<vector<int>> buff, vector<int> input, int k);

//Rect outputTargetImage(const FGObject& obj, InputArray src, InputArray fgSrc, OutputArray dst, OutputArray dstFg, Mat& R, Point& shift);

Point computeOrigCoord(Mat R, Point inputPt);

#endif
