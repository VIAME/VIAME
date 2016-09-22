#include "mex.h"
#include "matrix.h"
#include <iostream>
#include "cv.h"
#include "math_common.hpp"
//need cstdint for uint8_t
#include <cstdint>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/flann.hpp"
#include "highgui.h"

#include "util.h"
#include "classHierarchy.h"
#include "SpeciesIDLib.h"

using namespace std;

int main(cv::Mat img, cv:: Mat img2, int & pred, double & prob, Mat &fgRect)
{
    FishSpeciesID fishSpeciesID;
    
    //cv::vector<Mat> images;
    //images.push_back(img);
    //Display size of image, which appears to be opposite that of a 2D 
    //image in matlab- this doesn't work for opencv 2.3.1
    //cv::Size ss = img.size();
    //cout << ss;
    
    //Display image
    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE);// Create a window for display.
    cv::imshow( "Display window", img);
	
    //cout << "Reading the classifier model ... ";
	string xmlFilename = "Model_SVM.xml";//"hierPartClassifier.xml";
	fishSpeciesID.loadModel(xmlFilename.c_str());
	//cout << "done" << endl;
    
    //cout << "Predicting data ... " << endl;
    vector<int> predictions;
    vector<double> probabilities;
    
	//waitKey(0);
    bool isPartial = fishSpeciesID.predict(img, img2, predictions, probabilities, fgRect);
    
    pred=predictions.back();
    prob=probabilities.back();
    //cout << predictions.size();
    cout << prob;
    return (int)isPartial;
}

