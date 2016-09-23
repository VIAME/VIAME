#include <iostream>
#include <ctime>

#include "FGExtraction.h"

void FGExtraction::init()
{
	if(!_inImg.data) return;
	_fgImg = Mat::zeros(_inImg.size(), CV_8U);
	_bgsImg = Mat::zeros(_inImg.size(), CV_8U);
	_highMask = Mat::zeros(_inImg.size(), CV_8U);
	_lowMask = Mat::zeros(_inImg.size(), CV_8U);
	_fgHighImg = Mat::zeros(_inImg.size(), CV_8U);
	_fgLowImg = Mat::zeros(_inImg.size(), CV_8U);
	_ratioHistBP = Mat::zeros(_inImg.size(), CV_64F);
}

/*******************************************************************************
* Function:      extractFGTargets  
* Description:   extract FG targets with given conditions and return objects
* Arguments:
	inImg           -   input image
	fgImg           -   output FG mask image
	seLength        -   length of structuring elements (opening)
	threshVal       -   threshold value for converting to binary image
	minArea         -   minimum area of FG targets
	maxArea         -   maximum area of FG targets
	minAspRatio     -   minimum aspect ratio of FG targets
	maxAspRatio     -   maximum aspect ratio of FG targets
	
* Returns:       vector<FGObject>* - all extracted FG targets
* Comments:
* Revision: 
*******************************************************************************/
vector<FGObject>*
FGExtraction::extractFGTargets(InputArray inImg, OutputArray fgImg, int seLength, int threshVal, 
                                   double minArea, double maxArea, 
								   double minAspRatio, double maxAspRatio)
{
	double theta = 0.4;

	if(!inImg.obj) return NULL;

	_inImg = inImg.getMat();
	this->init();

	//showImage("inImg", _inImg);

	// background subtraction by opening
    int err = subtractBGOpenDiagonal(inImg, _bgsImg, threshVal, seLength);

	if (err>0) {
		vector<FGObject>* fgObjects = new vector<FGObject>;
		return fgObjects;
	}

	//subtractBGMedian(inImg.getMat(), bgSubImg, threshVal, seLength);
	//showImage("inImg", _inImg, 0, 1);
	//showImage("bgSub", _bgsImg);
	
    // get the contour
    vector<vector<Point>> contours = extractContours(_bgsImg);
	//cout<<contours.size()<<endl;

    // double local thresholding
    // histogram backprojection
    Mat mask = Mat::zeros(_bgsImg.size(), CV_8U); 
	vector<int> areas(contours.size());
	int cnt = 0;
	int argMax = 0;
	int max_area = 0;
    for(vector<vector<Point> >::const_iterator it = contours.begin(); it != contours.end(); ++it){
        Rect uprightBox = boundingRect(*it);
		areas[cnt] = uprightBox.height*uprightBox.width;
		if (areas[cnt]>max_area) {
			max_area = areas[cnt];
			argMax = cnt;
		}
		cnt++;
	}
	//showImage("inImg", _inImg, 0, 1);

	vector<Point> largestContour = contours[argMax];	//***** only use the largest contour
		RotatedRect orientedBox = orientedBoundingBox(largestContour);
		orientedBox.size.width *= 1.5;
        orientedBox.size.height *= 1.5;
		ellipse(mask, orientedBox, Scalar(255), -1);

		//Rect tempRect = boundingRect(largestContour);
		//Mat tempImg = mask(tempRect);
		//imshow("tempImg", tempImg);
		//imshow("mask", mask);
		//waitKey(0);

		// double local thresholding
		double percentage = 0.8;
		doubleThresholdByValue(percentage, mask);
		
		/*finish = clock();
		duration = (double)(finish - start) / (double)CLOCKS_PER_SEC;
		cout << duration << " sec" << endl;
		start = clock();*/

		// remove noise by a median filter
		medianBlur(_fgHighImg, _fgHighImg, 3);
		medianBlur(_fgLowImg, _fgLowImg, 3);
		//showImage("_fgHighImg", _fgHighImg);
		//showImage("_fgLowImg", _fgLowImg);

		/*finish = clock();
		duration = (double)(finish - start) / (double)CLOCKS_PER_SEC;
		cout << duration << " sec" << endl;
		
		start = clock();*/
		// merge two masks using histogram backprojection
		//showImage("_fgImg", _fgImg);
		//showImage("mask", mask);
		updateByHistBackproject(theta, mask);
		

		ellipse(mask, orientedBox, Scalar(0), -1);
		ellipse(_fgHighImg, orientedBox, Scalar(0), -1);
		ellipse(_fgLowImg, orientedBox, Scalar(0), -1);

    //}

	
    // thresholding by area and variance
#ifdef IMAGE_DOWNSAMPLING
	int dilateSESize = 3;
	int erodeSESize = 3;
	int varThresh = 30;
#else
	int dilateSESize = 7;
	int erodeSESize = 7;
	int varThresh = 30;
#endif

    //showImage("fg high", _fgHighImg, 0, 1);
    //showImage("fg low", _fgLowImg, 0, 1);
	//showImage("after histbp", _fgImg, 0);

	thresholdByAreaRatioVar(minArea, maxArea, dilateSESize, erodeSESize, minAspRatio, maxAspRatio, varThresh);
	
	//showImage("after area threshold", _fgImg, 0);

	// post-processing
    postProcessing(_fgImg, _fgImg);
	//imshow("_fgImg",_fgImg);
	//waitKey(0);

	// fill the holes of the fgImg
	_fgImg.copyTo(fgImg);
	floodFill(fgImg, cv::Point(0,0), Scalar(255));
	//imshow("fgImg",fgImg);
	//waitKey(0);

    bitwise_not(fgImg, fgImg);
	bitwise_or(fgImg, _fgImg, _fgImg);
	//imshow("inImg", inImg);
	//imshow("_fgImg",_fgImg);
	//waitKey(0);

	// opening
	RotatedRect rotatedR = fitEllipse(Mat(largestContour));
	float objHeight = min(rotatedR.size.height,rotatedR.size.width);
	int seSize = int(objHeight/10.0+0.5);
	
	Mat se = getStructuringElement(MORPH_ELLIPSE, Size(seSize, seSize));		//***** choose different size according to object height
	morphologyEx(_fgImg, _fgImg, MORPH_OPEN, se);

	//imshow("_fgImg",_fgImg);
	//waitKey(0);


	// close
	morphologyEx(_fgImg, _fgImg, MORPH_CLOSE, se);

	// timer
	/*clock_t start, finish;
	double duration = 0.0;
	start = clock();

	finish = clock();
	duration = (double)(finish - start) / (double)CLOCKS_PER_SEC;
	cout << duration << " sec" << endl;*/

	thresholdByAreaRatioVar(0.5*minArea, maxArea, 1, 1, minAspRatio, maxAspRatio, 30);

	// push targets into our vector
	//Mat largeInImg;
#ifdef IMAGE_DOWNSAMPLING
	resize(_fgImg, _fgImg, Size(), 2, 2, INTER_LINEAR);
	resize(_inImg, largeInImg, Size(), 2, 2, INTER_LINEAR);
#endif
	//tempImg = _fgImg.clone(); 
	//findContours(tempImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//tempImg.release();

	//imshow("_fgImg",_fgImg);
	//waitKey(0);
	contours = extractContours(_fgImg);

	vector<FGObject>* fgObjects = new vector<FGObject>;
	//Mat mask8U = Mat::zeros(largeInImg.size(), CV_8U); 
	for (size_t i = 0; i < contours.size(); i++){
		double area = contourArea(contours[i]);
		RotatedRect orientedRect = orientedBoundingBox(contours[i]);
		Point2f points[4];
		orientedRect.points(points);
		/*
		orientedRect.size.width *= 1.5;
        orientedRect.size.height *= 1.5;
		ellipse(mask8U, orientedRect, Scalar(255), -1);
		
		int channels[] = {0};
		int nbins = 16;
		const int histSize[] = {nbins};
		float range[] = {0, 255};
		const float* ranges[] = {range};
		Mat hist;
		cv::calcHist(&largeInImg, 1, channels, mask8U, hist, 1, histSize, ranges);
		*/
		// push targets into our vector
		FGObject* obj = new FGObject;
		//obj->histogram = hist;
		obj->setObjectProperties(area, orientedRect.angle, contours[i], points, SOURCE_UNRECTIFIED);

		if(obj->isPartialOut(_fgImg.cols, _fgImg.rows) == false){
			fgObjects->push_back(*obj);
		}
		delete obj;

		//ellipse(mask8U, orientedRect, Scalar(0), -1);
		
	}

	//  eliminate artifacts with width of 1 at the border...
	rectangle(_fgImg, Point(0,0), Point(_fgImg.cols-1, _fgImg.rows-1), Scalar(0));
	
	fgImg.getMatRef() = _fgImg.clone();
	return fgObjects;
}

/*******************************************************************************
* Function:      subtractBGOpenDiagonal  
* Description:   BG subtraction via opening with diagonal structuring elements
* Arguments:
	inImg           -   input image
	bgsImg          -   BG subtracted image
	threshVal       -   threshold value for converting to binary image
	seLength        -   length of structuring elements
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
int
FGExtraction::subtractBGOpenDiagonal(InputArray src, OutputArray dst, int threshVal, int seLength)
{
    // generate binary image by thresholding
	Mat bin;
	double thresh = threshold(src, bin, threshVal, 255, THRESH_BINARY);

	// opening by horizontal structuring element
	//Mat structElemHorizontal = Mat::ones(1, seLength, CV_8U);
	//morphologyEx(bin, dst, MORPH_OPEN, structElemHorizontal);

	// opening by vertical structuring element
	//Mat structElemVertical = Mat::ones(seLength, 1, CV_8U);
	//morphologyEx(dst, dst, MORPH_OPEN, structElemVertical);

	//imshow("src", src);
	//imshow("bin", bin);
	//waitKey(0);

    // opening by first diagonal structuring element
	Mat structElemBackSlash = Mat::eye(seLength, seLength, CV_8U);
	morphologyEx(bin, dst, MORPH_OPEN, structElemBackSlash);

	//imshow("dst1", dst);
	//waitKey(0);

    // opening by second diagonal structuring element
	Mat structElemSlash;
	flip(structElemBackSlash, structElemSlash, 0);
	morphologyEx(dst, dst, MORPH_OPEN, structElemSlash);

	//imshow("dst2", dst);
	//waitKey(0);

	// eliminate small noise
	Mat structElemEllip = getStructuringElement(MORPH_ELLIPSE, Size(seLength, seLength));
	morphologyEx(dst, dst, MORPH_OPEN, structElemEllip);

	//imshow("dst3", dst);
	//waitKey(0);


	// get object size
	Mat dstImg = dst.getMat();
	vector<vector<Point>> contours = extractContours(dstImg);
	if (contours.size()==0)
		return 1;

    Mat mask = Mat::zeros(_bgsImg.size(), CV_8U); 
	vector<int> areas(contours.size());
	int cnt = 0;
	int argMax = 0;
	int max_area = 0;
    for(vector<vector<Point> >::const_iterator it = contours.begin(); it != contours.end(); ++it){
        Rect uprightBox = boundingRect(*it);
		areas[cnt] = uprightBox.height*uprightBox.width;
		if (areas[cnt]>max_area) {
			max_area = areas[cnt];
			argMax = cnt;
		}
		cnt++;
	}
	vector<Point> largestContour = contours[argMax];	//***** only use the largest contour
	RotatedRect orientedBox = orientedBoundingBox(largestContour);

	int updateSeL = int(min(orientedBox.size.width, orientedBox.size.height)/5.0+0.5);

	// opening by first diagonal structuring element
	structElemBackSlash = Mat::eye(updateSeL, updateSeL, CV_8U);
	morphologyEx(bin, dst, MORPH_OPEN, structElemBackSlash);
	
	//imshow("dst1", dst);
	//waitKey(0);

    // opening by second diagonal structuring element
	flip(structElemBackSlash, structElemSlash, 0);
	morphologyEx(dst, dst, MORPH_OPEN, structElemSlash);

	//imshow("dst2", dst);
	//waitKey(0);

	// eliminate small noise
	structElemEllip = getStructuringElement(MORPH_ELLIPSE, Size(updateSeL, updateSeL));
	morphologyEx(dst, dst, MORPH_OPEN, structElemEllip);

	//imshow("dst3", dst);
	//waitKey(0);
	return 0;
}

/*******************************************************************************
* Function:      subtractBGMedian
* Description:   BG subtraction via opening with diagonal structuring elements
* Arguments:
	inImg           -   input image
	bgsImg          -   BG subtracted image
	threshVal       -   threshold value for converting to binary image
	seLength        -   length of structuring elements
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void
FGExtraction::subtractBGMedian(InputArray src, OutputArray dst, int threshVal, int seLength)
{
	Mat inImg = src.getMat();
	Mat medImg;
	
    // median filter
	Mat tempImg = inImg.clone();
	medianBlur(tempImg, medImg, 31);
	//showImage("median", medImg);
	
    Mat bin;
	double thresh = threshold(medImg, bin, threshVal, 255, THRESH_BINARY);
	
	dst.getMatRef() = bin;
}

/*******************************************************************************
* Function:      doubleThresholdByValue  
* Description:   performs double local thresholding using Otsu's method
* Arguments:
	percentage      -   parameter for amount of lowering Otsu threshold
	roiMask         -   ROI binary mask
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void
FGExtraction::doubleThresholdByValue(double percentage, Mat roiMask)
{
	int u = 0;
	int thresh = getOtsuThreshold(0, 255, &u, roiMask);
	int highThresh = thresh - int(percentage*(thresh - u));
	int lowThresh = thresh - int(1.4*percentage*(thresh - u));

#ifdef DEBUG
	//cout << "th_H = " << highThresh << "; th_L = " << lowThresh << "; \n";
#endif

	Mat maskedFG;
	bitwise_and(_inImg, roiMask, maskedFG);
	
	// generate high and low look-up tables
	Mat highLUT(1, 256, CV_8U);
	Mat lowLUT(1, 256, CV_8U);
    uchar* highData = highLUT.data; 
	uchar* lowData = lowLUT.data;
    for(int i = 0; i < 256; ++i){
        highData[i] = i <= highThresh ? 0 : 255;
		lowData[i] = i <= lowThresh ? 0 : 255;
	}

	// threshold by using look-up tables
	LUT(maskedFG, highLUT, _highMask);
	LUT(maskedFG, lowLUT, _lowMask);
	bitwise_or(_fgHighImg, _highMask, _fgHighImg, roiMask);
	bitwise_or(_fgLowImg, _lowMask, _fgLowImg, roiMask);
}

/*******************************************************************************
* Function:      getOtsuThreshold  
* Description:   computes the threhsold using Otsu's method
* Arguments:
	lowerVal      -   lower bound of pixel value
	upperVal      -   upper bound of pixel value
	u1Ptr         -   pointer to receive the mean of class 1
	roiMask       -   ROI binary mask
	
* Returns:       int - Otsu threshold
* Comments:
* Revision: 
*******************************************************************************/
inline int 
FGExtraction::getOtsuThreshold(int lowerVal, int upperVal, int* u1Ptr, InputArray roiMask)
{
	Mat _roiMask = roiMask.getMat();

	int channels[] = {0};
	int nbins = 256;
    const int histSize[] = {nbins};
    float range[] = {0, 255};
    const float* ranges[] = {range};
	Mat hist;
    cv::calcHist(&_inImg, 1, channels, roiMask, hist, 1, histSize, ranges);
	
	Mat_<float> hist_(hist);
	float size = float(sum(hist)[0]);

	float w1, w2, u1, u2;
  	float max = -1;
	int index = 1;
	float u1max = -1;
	float histMax = 0;
	int mode = 0;
	float count = 0;

	for (int i = lowerVal+1; i < upperVal; ++i){	
		if(hist_(i,0) > histMax) {
			histMax = hist_(i,0);
			mode = i;
		}
		w1 = 0;
		
		for (int j = lowerVal+1; j <= i; ++j){
			w1 = w1 + hist_(j-1,0);
		}
		w1 = w1 / size;
		w2 = 1 - w1;

		u1 = 0;
		count = 0;
		for (int j = lowerVal; j <= i-1; ++j){
			u1 = u1 + j*hist_(j,0);
			count += hist_(j,0);
		}
		u1 /= count;

		u2 = 0;
		count = 0;
		for (int j = i; j <= upperVal; ++j){
			u2 = u2 + j*hist_(j, 0);
			count += hist_(j, 0);
		}
		u2 /= count;

		if (w1 * w2 * (u1-u2) * (u1-u2) > max){
			max = w1 * w2 * (u1-u2) * (u1-u2);
			index = i;
			u1max = u1;
		}
		else{
			max = max;
			index = index;
		}
	}
	
	//cout << "mode = " << mode << endl;
	//cout << "u1 = " << u1max << "; index = " << index << "; ";
	
	*u1Ptr = (int)(u1max + 0.5);
	return index;
}

/*******************************************************************************
* Function:      updateByHistBackproject  
* Description:   merges low and high FG mask using histogram backprojection
* Arguments:
	theta         -   threshold for ratio histogram
	roiMask       -   ROI
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void
FGExtraction::updateByHistBackproject(double theta, Mat roiMask)
{
	// generate histograms of two foregrounds
    int channels[] = {0};
	int nbins = 16;
    const int histSize[] = {nbins};
    float range[] = {0, 255};
    const float* ranges[] = {range};
	Mat highHist;
    Mat lowHist;
    cv::calcHist(&_inImg, 1, channels, _highMask, highHist, 1, histSize, ranges);
	cv::calcHist(&_inImg, 1, channels, _lowMask, lowHist, 1, histSize, ranges);
	
	//cout << highHist << endl;
	//cout << lowHist << endl;

	// get the ratio histogram
	Mat ratioHist = highHist / lowHist;
	threshold(ratioHist, ratioHist, 1.0, 1.0, THRESH_TRUNC);

	// backproject the ratio histogram to image plane
	_ratioHistBP = histBackProject(ratioHist, roiMask);
	Mat hist;
	float range2[] = {0, 1.01};
    const float *ranges2[] = { range2 };
	const int histSize2[] = {256};
	cv::calcHist(&_ratioHistBP, 1, channels, Mat(), hist, 1, histSize2, ranges2, true, false);

	float totalPixelNum = _ratioHistBP.rows*_ratioHistBP.cols;
	float num_ratio = 0.2;
	float acc_sum = 0.0;
	float thresh = 0.4;
	for (int n = 0; n < 256; n++) {
		acc_sum += hist.at<float>(n);
		if (acc_sum>=num_ratio*totalPixelNum) {
			thresh = 1.01/256.0*n+1.01/256.0;
			break;
		}
	}


	// Show the calculated histogram in command window
    /*double total;
    total = _ratioHistBP.rows * _ratioHistBP.cols;
    for( int h = 0; h < 256; h++ )
         {
            float binVal = hist.at<float>(h);
            cout<<" "<<binVal;
         }

	// Plot the histogram
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/256 );
 
    Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
     
    for( int i = 1; i < 256; i++ )
    {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
    }
 
    namedWindow( "Result", 1 );    imshow( "Result", histImage );*/

	// thresholding on the backprojection	
	//imshow("_ratioHistBP",_ratioHistBP);
	//waitKey(0);
	threshold(_ratioHistBP, _ratioHistBP, thresh, 255, THRESH_BINARY);
	//imshow("_ratioHistBP",_ratioHistBP);
	//waitKey(0);

	Mat ucharBP;
	_ratioHistBP.convertTo(ucharBP, CV_8U);
	//imshow("ucharBP",ucharBP);
	//imshow("roiMask",roiMask);
	
	//showImage("fg", ucharBP);
	bitwise_or(ucharBP, _fgImg, _fgImg, roiMask);
	//imshow("_fgImg",_fgImg);
	//waitKey(0);

	//showImage("fg", _fgImg);
}

/*******************************************************************************
* Function:      myUpdateByHistBackproject  
* Description:   merges low and high FG mask using histogram backprojection
* Arguments:
	theta         -   threshold for ratio histogram
	roiMask       -   ROI
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void
FGExtraction::myUpdateByHistBackproject(double theta, Mat roiMask)
{
	int histSize = 16;
	vector<int> highHist;
    vector<int> lowHist;
	calcHist(_fgHighImg, highHist, histSize, roiMask);
	calcHist(_fgLowImg, lowHist, histSize, roiMask);
	
	vector<double> ratioHist;
	ratioHist.resize(histSize);
	for(int h = 0; h < histSize; ++h){
		ratioHist[h] = lowHist[h]==0 ? 0.0 : min( 1.0, highHist[h] / double(lowHist[h]) );
	}

	_ratioHistBP = histBackProject(ratioHist, roiMask);
	
	for(int y = 0; y < _ratioHistBP.rows; ++y){
		for(int x = 0; x < _ratioHistBP.cols; ++x){
			if(roiMask.at<uchar>(y, x) > 0){
				if(_fgLowImg.at<uchar>(y, x) > 0){
					if(_ratioHistBP.at<double>(y, x) >= theta)
						_fgImg.at<uchar>(y, x) = 255;
					else
						_fgImg.at<uchar>(y, x) = 0;
				}
			}
		}
	}
	
}

/*******************************************************************************
* Function:      calcHist  
* Description:   calculates the histogram
* Arguments:
	fg            -   FG mask
	hist          -   output histogram
	histSize      -   number of bins in histogram
	roiMask       -   ROI mask
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void
FGExtraction::calcHist(Mat fg, vector<int>& hist, int histSize, Mat roiMask)
{
	hist.resize(histSize, 0);
	int binWidth = 256 / histSize;
	
    for (int y = 0; y < _inImg.rows; ++y){
        for (int x = 0; x < _inImg.cols; ++x){
			if(roiMask.at<uchar>(y, x) > 0 && fg.at<uchar>(y, x) > 0){
				int idx = _inImg.at<uchar>(y, x) / binWidth;
				++hist[idx];
			}
		}
	}
}

/*******************************************************************************
* Function:      histBackProject  
* Description:   performs histogram backprojection
* Arguments:
	hist          -   histogram
	histSize      -   number of bins in histogram
	roiMask       -   ROI mask
	
* Returns:       Mat - the histogram backproject image
* Comments:
* Revision: 
*******************************************************************************/
Mat
FGExtraction::histBackProject(InputArray hist, Mat roiMask)
{
	Mat histMat = hist.getMat();
	int binWidth = histMat.rows;

	// generate high and low look-up tables
	Mat lookUpTable(1, 256, CV_32F);
    for(int i = 0; i < 256; ++i)
        lookUpTable.at<float>(0, i) = histMat.at<float>(i/binWidth, 0);

	//cout << lookUpTable << endl;
	//system("Pause");

	// perform histogram backprojection via the look-up table
	Mat backProj = Mat(_inImg.size(), CV_32F);
	LUT(_inImg, lookUpTable, backProj);
	
	/*
    for (int y = 0; y < _inImg.rows; ++y){
        for (int x = 0; x < _inImg.cols; ++x){
			if(roiMask.at<uchar>(y, x) > 0){
				int idx = _inImg.at<uchar>(y, x) / binWidth;
                float bp = histMat.at<float>(idx, 0);
                backProj.at<float>(y, x) = bp;
			}
		}
	}
	*/

	return backProj;
}

/*******************************************************************************
* Function:      thresholdByAreaRatioVar  
* Description:   thresholds the FG targets by area, aspect ratio and variance
* Arguments:
	minArea       -   minimum area to preserve
	maxArea       -   maximum area to preserve
	dilateSESize  -   structuring element size for dilation
	erodeSESize   -   structuring element size for erosion
	minAspRatio   -   minimum aspect ratio
	maxAspRatio   -   maximum aspect ratio
	varThresh     -   value of variance threshold
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void
FGExtraction::thresholdByAreaRatioVar(double minArea, double maxArea, 
                                          int dilateSESize, int erodeSESize, 
                                          double minAspRatio, double maxAspRatio, int varThresh)
{
	bool passArea, passRatio, passVar;
	vector<vector<Point> > contours;

	// connect separate parts before finding connected components
	Mat dilateSE = getStructuringElement(MORPH_ELLIPSE, Size(dilateSESize, dilateSESize));
	Mat erodeSE = getStructuringElement(MORPH_ELLIPSE, Size(erodeSESize, erodeSESize));
	dilate(_fgImg, _fgImg, dilateSE);
	erode(_fgImg, _fgImg, erodeSE);

	// extract contours of targets
    Mat tempImg = _fgImg.clone();
    findContours(tempImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
    tempImg.release();

	for(size_t i = 0; i < contours.size(); ++i){
		passArea = passRatio = passVar = false;
		double area = contourArea(contours[i]);
		
		// check if the area is within the desired range
		if (contours[i][0].y > 0.35*_fgImg.rows && contours[i][0].y < 0.68*_fgImg.rows && area > 0.1*minArea && area < maxArea
			|| area > minArea && area < maxArea) 
			passArea = true;
		

		// check if the aspect ratio is within the desired range
		RotatedRect orientedRect = orientedBoundingBox(contours[i]);
		Point2f rectPoints [4]; orientedRect.points(rectPoints);
		double rectWidth = norm(rectPoints[0] - rectPoints[1]);
		double rectHeight = norm(rectPoints[1] - rectPoints[2]);
		double aspRatio = max(rectWidth / rectHeight, rectHeight / rectWidth);
		if (aspRatio > minAspRatio && aspRatio < maxAspRatio) passRatio = true;

		// check if the variance of pixel exceeds the threshold
		// TODO ...
		passVar = true;

		// remove the target if any of the tests fails
		if(!passArea || !passRatio || !passVar){
			//Rect uprightRect = boundingRect(Mat(contours[i]));
			//rectangle(_fgImg, uprightRect, Scalar(0), -1);
			drawContours(_fgImg, contours, i, Scalar(0), -1);
		}
	}

	/*namedWindow("fgImg", 0);
	imshow("fgImg", _fgImg);
	waitKey(0);*/
}

/*******************************************************************************
* Function:      postProcessing  
* Description:   performs post processing with morphology
* Arguments:
	src           -   input image
	dst           -   output image
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void 
FGExtraction::postProcessing(InputArray src, OutputArray dst)
{
	Mat _src = src.getMat();
	// 1. close --> open --> close
	int closeSEsize = 5;
	int openSEsize = 5;

	Mat tempImg;

	Mat closeElement = getStructuringElement(MORPH_ELLIPSE, Size(closeSEsize, closeSEsize));
	Mat openElement = getStructuringElement(MORPH_ELLIPSE, Size(openSEsize, openSEsize));

	morphologyEx(_src, tempImg, MORPH_CLOSE, closeElement);
	morphologyEx(tempImg, tempImg, MORPH_OPEN, openElement);
	morphologyEx(tempImg, dst, MORPH_CLOSE, closeElement);

	// 2. dilate --> erode
	/*int dilateSEsize = 7;
	IplConvKernel* addHorLine = cvCreateStructuringElementEx(dilateSEsize, dilateSEsize, dilateSEsize/2, dilateSEsize/2, CV_SHAPE_RECT );
	cvDilate(tempImg, tempImg, addHorLine);
	cvReleaseStructuringElement(&addHorLine);

	int erodeSEsize = 7;
	IplConvKernel* removeHorLine = cvCreateStructuringElementEx(erodeSEsize, erodeSEsize, erodeSEsize/2, erodeSEsize/2, CV_SHAPE_RECT );
	cvErode(tempImg, outImg, removeHorLine);
	cvReleaseStructuringElement(&removeHorLine);*/
}

/*******************************************************************************
* Function:      medianFilter  
* Description:   performs median filter with kernel size kwidth*kheight
* Arguments:
	src           -   input image
	dst           -   output image
	kwidth        -   
	kheight       -   
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void FGExtraction::medianFilter(Mat src, Mat dst, int ksize, bool isHorizontal)
{
	assert(ksize % 2 == 1);
	dst = src;
	int kwidth = isHorizontal ? ksize : 1;
	int kheight = isHorizontal ? 1 : ksize;

	for(int y = kheight/2; y < src.rows-kheight/2; ++y){
		for(int x = kwidth/2; x < src.cols-kwidth/2; ++x){
			vector<uchar> vec;
			vec.reserve(kwidth);
			uchar* p = 0;
			for(int v = -kheight/2; v <= kheight/2; ++v){
				p = src.ptr(y+v);
				for(int u = -kwidth/2; u <= kwidth/2; ++u)
					vec.push_back(p[x+u]);
			}
				
			sort(vec.begin(), vec.end());
			p = src.ptr(y);
			p[x] = vec[ksize/2];
		}
	}
}