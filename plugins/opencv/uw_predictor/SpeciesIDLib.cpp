//////////////////////////////////////////////////////////////////////////
//
//  UWEESpeciesIDLib.cpp
//  Date:   Jun/12/2014
//  Author: Meng-Che Chuang, University of Washington
//
//  Implementation file for species classifier
//
//////////////////////////////////////////////////////////////////////////

#include "util.h"
#include "SpeciesIDLib.h"

#include <fstream>

int FishSpeciesID::_dimFeat = 131;

//********** class FishSpeciesID *************************************************
FishSpeciesID::FishSpeciesID() : _classHierarchy(ClassHierarchy()), _count(0)
{
	// parameter default values
	_thresh = 48;
	_seLength = 5; //7;
	_minArea = 200;
	_maxArea = 1e6;
	_minAspRatio = 1.8;
	_maxAspRatio = 8.0;
	_histBP_theta = 0.2;
}

FishSpeciesID::~FishSpeciesID()
{

}

void FishSpeciesID::loadModel(const char* filename)
{
	_classHierarchy.loadModel(filename);
}

void FishSpeciesID::saveModel(const char* filename)
{
	_classHierarchy.saveModel(filename);
}

int FishSpeciesID::outputFeature(Mat img, Mat img2, Mat& feature) {
	Mat fgRect(1,8,CV_64F);
	Point shift;
	Mat rotateR;
	int err = extractFeatures(img, img2, feature, false, true, fgRect, shift, rotateR);
	return err;
}


void FishSpeciesID::train(Mat data, Mat labels)
{
	// change double to float
	Mat trainData(data.rows, data.cols, CV_32FC1);
	Mat trainLabel(data.rows, 1, CV_32FC1);
	for (int i = 0; i < data.rows; i++)
		for (int j = 0; j < data.cols; j++) {
			trainData.at<float>(i,j) = (float)(data.at<double>(i,j));
			//if (i==0)
				//cout<<trainData.at<float>(i,j)<<endl;
		}
	for (int i = 0; i < labels.rows; i++)
		for (int j = 0; j < labels.cols; j++) {
			trainLabel.at<float>(i,j) = (float)(labels.at<double>(i,j));
			//cout<<trainLabel.at<float>(i,j)<<endl;
		}
	_classHierarchy.train(trainData, trainLabel);
	_classHierarchy.saveModel("Model_SVM.xml");
	return;
	/*int N = img_name1.size();
	Mat trainData;
	Mat trainLabels;
	Mat fgRect = Mat::zeros(1,8,CV_64F);
	Point shift;
	Mat rotateR;
	for (int n = 0; n < N; n++) {
		Mat img1 = imread(img_name1[n]);
		Mat img2 = imread(img_name2[n]);
		Mat feature(1,_dimFeat,CV_32F);
		int err = extractFeatures(img1, img2, feature, false, true, fgRect, shift, rotateR);
		if (err==0) {
			trainData.push_back(feature);
			trainLabels.push_back(labels[n]);
		}
	}
	_classHierarchy.train(trainData, trainLabels);
	_classHierarchy.saveModel("Model_SVM.xml");*/
}

bool FishSpeciesID::predict(Mat img, Mat img2, vector<int>& predictions, vector<double>& probabilities, Mat& fgRect)
{
	//predictions.clear();
	//probabilities.clear();

	++_count;

	Mat sample(1, _dimFeat, CV_32F);
	//cout<<_dimFeat<<endl;
	//waitKey(0);
	Point shift;
	Mat rotateR;
	int err = extractFeatures(img, img2, sample, false, true, fgRect, shift, rotateR);

	cout<<err<<endl;
	//waitKey(0);
	//cout<<sample.at<float>(0,100)<<endl;
	//cout<<sample.at<float>(0,104)<<endl;
	/*ofstream fout("features.csv", ios::app);
	fout << _count << ',';
	for (int i = 0; i < _dimFeat; ++i) {
		fout << setprecision(8) << fixed << sample.at<float>(0, i) << ',';
	}
	fout << endl;
	fout.close();*/

	bool isPartial = false;
	if (err==0) {
		isPartial = _classHierarchy.predict(sample, predictions, probabilities);
		//for (int kk = 0; kk < probabilities.size(); kk++)
			//cout<<probabilities[kk]<<endl;
		//cout<<predictions.back()<<endl;
		//cout<<predictions[0]<<endl;
		vector<int> class_label;
		vector<vector<double>> probb;
		bool temp = _classHierarchy.predict2(sample, class_label, probb);

		int pred = predictions.back();
		//cout<<pred<<endl;
		predictions.clear();
		probabilities.clear();
		predictions.push_back(pred);

		vector<double> final_prob(class_label.size(),1.0);
		for (int kk = 0; kk < class_label.size(); kk++) {
			for (int mm = 0; mm < probb[kk].size(); mm++) {
				final_prob[kk] = final_prob[kk]*probb[kk][mm];
			}
			//cout<<final_prob[kk]<<endl;
			if (class_label[kk]==pred) {
				probabilities.push_back(final_prob[kk]);
				//cout<<final_prob[kk]<<endl;
				break;
			}
		}
		if (probabilities.size()==0)
			probabilities.push_back(0.0);

		if (predictions.back()<0) {
			isPartial = true;
			return isPartial;
		}

		// measure length
		vector<Point> corners(4);	// order: toptail,head,head,bottail
		bool isTailAtRight;
		if (fgRect.at<double>(0,0)>fgRect.at<double>(0,2))
			isTailAtRight = true;
		else
			isTailAtRight = false;

		float tail_length = abs(fgRect.at<double>(0,0)-fgRect.at<double>(0,2))*(1-body_ratio[predictions.back()]);
		for (int k = 0; k < 4; k++) {
			corners[k].x = fgRect.at<double>(0,k*2);
			corners[k].y = fgRect.at<double>(0,k*2+1);
			if (k==0 || k==3) {
				if (isTailAtRight)
					corners[k].x += tail_length;
				else
					corners[k].x -= tail_length;
			}
			Point out_corner = computeOrigCoord(rotateR, corners[k]);
			fgRect.at<double>(0,k*2) = out_corner.x+shift.x;
			fgRect.at<double>(0,k*2+1) = out_corner.y+shift.y;
			if (fgRect.at<double>(0,k*2)<0)
				fgRect.at<double>(0,k*2) = 0.0;
			if (fgRect.at<double>(0,k*2)>img.cols-1)
				fgRect.at<double>(0,k*2) = img.cols-1;
			if (fgRect.at<double>(0,k*2+1)<0)
				fgRect.at<double>(0,k*2+1) = 0.0;
			if (fgRect.at<double>(0,k*2+1)>img.rows-1)
				fgRect.at<double>(0,k*2+1) = img.rows-1;
		}
	}
	else {
		isPartial = true;
		predictions.push_back(0);
		probabilities.push_back(0.0);
	}

	return isPartial;
}

int FishSpeciesID::extractFeatures(Mat src, Mat src2, Mat& features, bool useTailFD, bool useHeadFD, Mat &fgRect, Point& shift, Mat& rotateR)
{

	FeatureExtraction	featExtractor;

	// extract targets and generate their features
	FGExtraction fgExtractor;
	//for (int n = 0; n < numData; n++) {

		Mat fg;
		if(src.data == NULL) return 1;

		// perform segmentation for every target
		double thres_val = cv::threshold(src2, fg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		int		thresh = min(5,int(thres_val));//48;
		int		seLength = int(src.rows/20);//9;
		double	minArea = 200;
		double	maxArea = 1e6;
		double	minAspRatio = 1.8;
		double	maxAspRatio = 8.0;

		// segmentation by calling foreground extraction function
		vector<FGObject>* objects;
		//if (n==12)
		//	int aa = 0;
		//imshow("src2",src2);
		//waitKey(0);

		objects = fgExtractor.extractFGTargets(src2, fg, seLength, thresh, minArea, maxArea, minAspRatio, maxAspRatio);

		//imshow("fg",fg);
		//waitKey(0);
		//cout<<(*objects).size()<<endl;
		if ((*objects).size()==0)
			return 2;

		vector<vector<Point>> orig_contours = extractContours(fg);
		RotatedRect tempRect = orientedBoundingBox(orig_contours[0]);
		/*fgRect.at<double>(0,0) = tempRect.center.x;
		fgRect.at<double>(0,1) = tempRect.center.y;
		fgRect.at<double>(0,2) = tempRect.size.width;
		fgRect.at<double>(0,3) = tempRect.size.height;
		fgRect.at<double>(0,4) = tempRect.angle;*/

		//imshow("fg2",fg);
		//waitKey(0);



		// preserve the one with largest area
		int idx = -1;
		double maxObjArea = 0;
		for(size_t k = 0; k < objects->size(); ++k){
			double area = (*objects)[k].area;
			if(area > maxObjArea){
				idx = k;
				maxObjArea = area;
			}
		}

		if(idx == -1) idx = 0;
		FGObject obj = (*objects)[idx];

		/*****************************************************************************/
		//imshow("src",src);
		//waitKey(0);
		// rectify images in horizontal direction
		Mat targetImg, targetFgImg;//, rotateR;
		//Point shift;
		Rect roiRect = outputTargetImage(obj, src, fg, targetImg, targetFgImg, rotateR, shift);
		//imshow("src",src);
		//imshow("fg",fg);
		//imshow("targetImg",targetImg);
		//imshow("targetFgImg",targetFgImg);
		//waitKey(0);

		// get grad hist
		vector<double> grad_hist = getGradHist(targetImg, targetFgImg);

		vector<vector<Point>> contours = extractContours(targetFgImg);

		Mat tempFg = Mat::zeros(targetImg.size(), targetImg.type());
		if (idx>contours.size()-1)
			idx = 0;

		drawContours(tempFg, contours, idx, Scalar(255), -1);
		//imshow("tempFg",tempFg);
		//waitKey(0);
		targetFgImg = tempFg;

		vector<Point> targetContour = contours[idx];


		Mat targetImgC3;
		cvtColor(targetImg, targetImgC3, COLOR_GRAY2BGR);
		Mat targetFgImgC3;
		cvtColor(targetFgImg, targetFgImgC3, COLOR_GRAY2BGR);

		int radius = int(obj.uWidth / 80.0);

		Rect boundingBox = boundingRect(targetContour);
		Point2f boxCenter = 0.5*boundingBox.tl() + 0.5*boundingBox.br();

		//imshow("targetImgC3",targetImgC3);
		//imshow("targetFgImg",targetFgImg);
		//waitKey(0);

		// exporting images and segmentations to the folder "dataset"
		//sprintf_s(imgPath, "%s\\images\\%s\\%03d.jpg", param.getResultPath(), param.getClassLabel(i), j + 1);
		//imwrite(imgPath, targetImgC3);
		//sprintf_s(imgPath, "%s\\segmentations\\%s\\%03d.jpg", param.getResultPath(), param.getClassLabel(i), j + 1);
		//imwrite("C:\\Users\\ipl333\\Documents\\Code\\CamTrawl\\Image Dataset\\img2\\Rockfish_seg\\"+n+".png", targetFgImg);

		/*****************************************************************************/
		// decide the tail side by vertical projection
		vector<int> vertProj;
		vertProj.reserve(targetFgImg.cols);
		vector<int> derivVertProj;
		derivVertProj.reserve(targetFgImg.cols);

		float fishLen = max(boundingBox.width, boundingBox.height);
		int leftBound = int(boxCenter.x - 0.4*fishLen);
		int rightBound = int(boxCenter.x + 0.4*fishLen);
		int leftInnerBound = int(boxCenter.x - 0.2*fishLen);
		int rightInnerBound = int(boxCenter.x + 0.2*fishLen);
		int minSum = targetFgImg.rows;
		int argMinSum = 0;
		int maxSum = 0;
		int argMaxSum = 0;

		Mat vertProjImg = Mat::zeros(targetFgImgC3.size(), targetFgImgC3.type());
		Mat derivVertProjImg = Mat::zeros(targetFgImgC3.size(), targetFgImgC3.type());

		for(int x = 0; x < targetFgImg.cols; ++x){
			int sum = 0;
			for(int y = 0; y < targetFgImg.rows; ++y){
				if(x == leftBound || x == rightBound)
					vertProjImg.at<Vec3b>(y, x) = Vec3b(0, 255, 0);
				if(targetFgImg.at<uchar>(y, x))
					++sum;
			}
			vertProj.push_back(sum);
			if(x > leftBound && x < leftInnerBound || x > rightInnerBound && x < rightBound){
				if(sum < minSum) {
					minSum = sum;
					argMinSum = x;
				}
				if(sum > maxSum) {
					maxSum = sum;
					argMaxSum = x;
				}
			}

			int yOnPlot = targetFgImg.rows - sum;
			if(yOnPlot < targetFgImg.rows)
				circle(vertProjImg, Point(x, yOnPlot), 1, Scalar::all(255), -1);

		}

		circle(vertProjImg, Point(argMinSum, targetFgImg.rows - vertProj[argMinSum]), 5, Scalar(0, 0, 255), -1);
		//imshow("vertProjImg",vertProjImg);
		//waitKey(0);

		int tailPointX = argMinSum;
		bool isTailAtRight = tailPointX > boxCenter.x;
		line(targetFgImgC3, Point(tailPointX, targetFgImg.rows / 4), Point(tailPointX, targetFgImg.rows * 3 / 4), Scalar(0, 0, 255), 3);
		//imshow("targetFgImgC3",targetFgImgC3);
		//waitKey(0);


		// find contour pts
		int contourNum = targetContour.size();
		Point leftPt = targetContour[0], rightPt = targetContour[0];
		Point headPt, topTailPt, botTailPt;
		for (int nn = 0; nn < contourNum; nn++) {
			if (targetContour[nn].x<leftPt.x)
				leftPt = targetContour[nn];
			if (targetContour[nn].x>rightPt.x)
				rightPt = targetContour[nn];
		}
		if (isTailAtRight)
			headPt = leftPt;
		else
			headPt = rightPt;
		topTailPt.x = tailPointX;
		topTailPt.y = 10000;
		botTailPt.x = tailPointX;
		botTailPt.y = 0;
		for (int nn = 0; nn < contourNum; nn++) {
			if (targetContour[nn].x==tailPointX) {
				if (targetContour[nn].y<topTailPt.y)
					topTailPt.y = targetContour[nn].y;
				if (targetContour[nn].y>botTailPt.y)
					botTailPt.y = targetContour[nn].y;
			}
		}
		double bodyLength = abs(headPt.x-tailPointX)+1;
		vector<vector<int>> ptsData(bodyLength);
		vector<int> bodyHeight(bodyLength);
		for (int nn = 0; nn < contourNum; nn++) {
			if (isTailAtRight) {
				if (targetContour[nn].x>=headPt.x && targetContour[nn].x<=tailPointX)
					ptsData[targetContour[nn].x-headPt.x].push_back(targetContour[nn].y);
			}
			else {
				if (targetContour[nn].x<=headPt.x && targetContour[nn].x>=tailPointX)
					ptsData[targetContour[nn].x-tailPointX].push_back(targetContour[nn].y);
			}
		}
		for (int nn = 0; nn < bodyLength; nn++) {
			if (ptsData[nn].size()==0)
				bodyHeight[nn] = 0;
			else {
				int sizeNum = ptsData[nn].size();
				int rowMin = ptsData[nn][0];
				int rowMax = ptsData[nn][0];
				for (int mm = 0; mm < sizeNum; mm++) {
					if (ptsData[nn][mm]<rowMin)
						rowMin = ptsData[nn][mm];
					if (ptsData[nn][mm]>rowMax)
						rowMax = ptsData[nn][mm];
				}
				bodyHeight[nn] = abs(rowMin-rowMax);
			}
		}
		int argMaxHeight = 0, maxHeight = bodyHeight[0];
		for (int nn = 0; nn < bodyLength; nn++) {
			if (bodyHeight[nn]>maxHeight) {
				maxHeight = bodyHeight[nn];
				argMaxHeight = nn;
			}
		}
		if (isTailAtRight)
			argMaxHeight += headPt.x;
		else
			argMaxHeight += tailPointX;

		//////////////////////////////////////////////////////////////////////////
		int minDiff = targetFgImg.rows;
		int argMinDiff = 0;
		leftInnerBound = int(boxCenter.x - 0.3*fishLen);
		rightInnerBound = int(boxCenter.x + 0.3*fishLen);

		vector<int> smoothedVertProj;
		smoothedVertProj.resize(vertProj.size(), 0);

		for(int x = 0; x < targetFgImg.cols; ++x){
			int sum = 0;
			for(int k = -5; k <= 5; ++k){
				int idx = x+k < 0 ? 0 : (x+k > targetFgImg.cols-1 ? targetFgImg.cols-1 : x+k);
				sum += vertProj[idx];
			}
			smoothedVertProj[x] = sum/11;
		}

		for(int x = 1; x < targetFgImg.cols; ++x){
			for(int y = 0; y < targetFgImg.rows; ++y){
				if(x == leftBound || x == rightBound)
					derivVertProjImg.at<Vec3b>(y, x) = Vec3b(0, 255, 0);
			}

			int diff = smoothedVertProj[x] - smoothedVertProj[x-1];

			derivVertProj.push_back(diff);
			if(isTailAtRight && x > leftBound && x < leftInnerBound){
				if(abs(diff) <= minDiff) {
					minDiff = diff;
					argMinDiff = x;
				}
			}
			else if(!isTailAtRight && x > rightInnerBound && x < rightBound){
				if(abs(diff) < minDiff) {
					minDiff = diff;
					argMinDiff = x;
				}
			}

			int yOnPlot = targetFgImg.rows/2 - diff*3;
			if(yOnPlot < targetFgImg.rows)
				circle(derivVertProjImg, Point(x, yOnPlot), 1, Scalar::all(255), -1);

		}
		circle(derivVertProjImg, Point(argMinDiff, targetFgImg.rows/2 - derivVertProj[argMinDiff]), 5, Scalar(255, 192, 0), -1);
		//imshow("derivVertProjImg",derivVertProjImg);
		//waitKey(0);
		/*
		Mat bodyImg;
		bitwise_and(targetImg, targetFgImg, bodyImg);
		Mat edgeImg = Mat::zeros(targetFgImgC3.size(), targetFgImgC3.type());
		Canny(bodyImg, edgeImg, 180, 120);
		*/

		// get bounding box for fish body only
		vector<Point> bodyContour;
		if (isTailAtRight) {
			for (int nn = 0; nn < contourNum; nn++) {
				if (targetContour[nn].x<=topTailPt.x)
					bodyContour.push_back(targetContour[nn]);
			}
		}
		else {
			for (int nn = 0; nn < contourNum; nn++) {
				if (targetContour[nn].x>=topTailPt.x)
					bodyContour.push_back(targetContour[nn]);
			}
		}
		Rect bbox_body = boundingRect(bodyContour);
		vector<Point> corners(4);
		// order: toptail,head,head,bottail
		if (isTailAtRight) {
			corners[0].x = bbox_body.x+bbox_body.width;
			corners[0].y = bbox_body.y;
			corners[1].x = bbox_body.x;
			corners[1].y = bbox_body.y;
			corners[2].x = bbox_body.x;
			corners[2].y = bbox_body.y+bbox_body.height;
			corners[3].x = bbox_body.x+bbox_body.width;
			corners[3].y = bbox_body.y+bbox_body.height;
		}
		else {
			corners[0].x = bbox_body.x;
			corners[0].y = bbox_body.y;
			corners[1].x = bbox_body.x+bbox_body.width;
			corners[1].y = bbox_body.y;
			corners[2].x = bbox_body.x+bbox_body.width;
			corners[2].y = bbox_body.y+bbox_body.height;
			corners[3].x = bbox_body.x;
			corners[3].y = bbox_body.y+bbox_body.height;
		}
		//vector<Point> output_corners(4);
		for (int nn = 0; nn < 4; nn++) {
			//output_corners[nn] = computeOrigCoord(rotateR, corners[nn]);
			//output_corners[nn].x += shift.x;
			//output_corners[nn].y += shift.y;
			fgRect.at<double>(0,nn*2) = corners[nn].x;
			fgRect.at<double>(0,nn*2+1) = corners[nn].y;
		}


		int neckPointX = argMinDiff;
		//neckPointX = int(boxCenter.x + (isTailAtRight ? -0.3 : 0.3)*fishLen);
		line(targetFgImgC3, Point(neckPointX, targetFgImg.rows / 4), Point(neckPointX, targetFgImg.rows * 3 / 4), Scalar(255, 192, 0), 3);
		//imshow("targetFgImgC3",targetFgImgC3);
		//waitKey(0);


		//sprintf_s(imgPath, "%s\\%s\\vertProj-%03d.jpg", param.getResultPath(), param.getClassLabel(i), j + 1);
		//imwrite(imgPath, vertProjImg);
		//sprintf_s(imgPath, "%s\\%s\\derivVertProj-%03d.jpg", param.getResultPath(), param.getClassLabel(i), j + 1);
		//imwrite(imgPath, derivVertProjImg);

		Mat srcColor;
		cvtColor(src, srcColor, COLOR_GRAY2RGB);
		Mat fgColor;
		cvtColor(fg, fgColor, COLOR_GRAY2RGB);



		//stringstream convert;
		//convert<<n;
		//char filename1[256], filename2[256], filename3[256];
		//sprintf(filename1, "C:\\Users\\ipl333\\Documents\\Code\\CamTrawl\\Image Dataset\\seg_result3\\%d_targetImg.jpg", n);
		//sprintf(filename2, "C:\\Users\\ipl333\\Documents\\Code\\CamTrawl\\Image Dataset\\seg_result3\\%d.png", n);
		//sprintf(filename3, "C:\\Users\\ipl333\\Documents\\Code\\CamTrawl\\Image Dataset\\seg_result3\\%d_vertProjImg.jpg", n);
		//imwrite(filename1, targetImg);
		//imwrite(filename2, targetFgImg);
		//imwrite(filename3, vertProjImg);
		//continue;
		/*****************************************************************************/
		// generate features for the target
		double objArea = sum(targetFgImg)[0]/255.0;

		// write feature: tail fin area and shape
		vector<Point> tailContour;

		for(int u = 0; u < targetContour.size(); ++u){
			if(isTailAtRight && targetContour[u].x > tailPointX || !isTailAtRight && targetContour[u].x < tailPointX)
				tailContour.push_back(targetContour[u]);
		}

		vector<Point> headContour;

		for(int u = 0; u < targetContour.size(); ++u){
			if(isTailAtRight && targetContour[u].x < neckPointX || !isTailAtRight && targetContour[u].x > neckPointX)
				headContour.push_back(targetContour[u]);
		}

		if (tailContour.size()==0 || headContour.size()==0)	//******
			return 3;

		headContour.clear();
		for(int u = 0; u < targetContour.size(); ++u){
			//****if(isTailAtRight && targetContour[u].x < neckPointX || !isTailAtRight && targetContour[u].x > neckPointX)
			//****	headContour.push_back(targetContour[u]);
			if (isTailAtRight) {
				if ((float)abs(targetContour[u].x-headPt.x)/(float)bodyLength<0.25 && (float)(targetContour[u].x-headPt.x)/(float)(argMaxHeight-headPt.x)<0.5)
					headContour.push_back(targetContour[u]);
			}
			else {
				if ((float)abs(targetContour[u].x-headPt.x)/(float)bodyLength<0.25 && (float)(headPt.x-targetContour[u].x)/(float)(headPt.x-argMaxHeight)<0.5)
					headContour.push_back(targetContour[u]);
			}
		}

		vector<Point> middleContour;
		for(int u = 0; u < targetContour.size(); ++u){
			if ((float)abs(targetContour[u].x-headPt.x)/(float)bodyLength<0.5 && (float)abs(targetContour[u].x-headPt.x)/(float)bodyLength>0.25)
				middleContour.push_back(targetContour[u]);
		}


		// record contour info
		int botTailIdx, headIdx;
		for (int u = 0; u < contourNum; u++) {
			if (targetContour[u].x==botTailPt.x && targetContour[u].y==botTailPt.y)
				botTailIdx = u;
			if (targetContour[u].x==headPt.x && targetContour[u].y==headPt.y)
				headIdx = u;
		}
		int currentCnt1 = botTailIdx;
		int currentCnt2 = botTailIdx;
		vector<Point> contour1, contour2;
		float y_avg1 = 0.0, y_avg2 = 0.0;
		while (currentCnt1!=headIdx) {
			contour1.push_back(targetContour[currentCnt1]);
			y_avg1 += contour1.back().y;
			currentCnt1 = (currentCnt1+1)%contourNum;
		}
		y_avg1 = y_avg1/(float)contour1.size();
		while (currentCnt2!=headIdx) {
			contour2.push_back(targetContour[currentCnt2]);
			y_avg2 += contour2.back().y;
			currentCnt2 = (currentCnt2-1+contourNum)%contourNum;
		}
		y_avg2 = y_avg2/(float)contour2.size();

		int nP = 128;
		vector<float> botContour(nP);
		if (y_avg1>y_avg2) {
			size_t size = contour1.size();
			for(int u = 0; u < nP; ++u)
				botContour[u] = (contour1[u*size/nP].y-y_avg1)/(float)maxHeight;
		}
		else {
			size_t size = contour2.size();
			for(int u = 0; u < nP; ++u)
				botContour[u] = (contour2[u*size/nP].y-y_avg2)/(float)maxHeight;
		}

		//if (objArea<1500)
		//	continue;

		double tailArea = contourArea(tailContour);
		//fout << tailArea / objArea << ',';
		//*****fout << tailArea / obj.area << ',';

		int featCnt = 0;
		if (useTailFD) {
			vector<double> tailFD;
			tailFD = featExtractor.getFourierDescriptor(tailContour);
			for(size_t k = 0; k < tailFD.size(); ++k){
				features.at<float>(0,featCnt) = tailFD[k];
				featCnt++;
			}
		}

		Rect tailRect = boundingRect(tailContour);
		if (~isTailAtRight) {
			tailRect.x -= int(0.25*bodyLength);
			if (tailRect.x<0)
				tailRect.x = 0;
			tailRect.y = topTailPt.y-int(0.25*maxHeight);
			if (tailRect.y<0)
				tailRect.y = 0;
			tailRect.width = abs(topTailPt.x-tailRect.x);
			tailRect.height = int(0.5*maxHeight)+botTailPt.y-topTailPt.y;
			if (tailRect.height+tailRect.y>=targetFgImg.rows)
				tailRect.height = targetFgImg.rows-tailRect.y-1;
		}
		else {
			tailRect.x = topTailPt.x;
			tailRect.y = topTailPt.y;
			if (tailRect.y<0)
				tailRect.y = 0;
			tailRect.width = int(0.25*bodyLength);
			if (tailRect.x+tailRect.width>=targetFgImg.cols)
				tailRect.width = targetFgImg.cols-tailRect.x-1;
			tailRect.height = int(0.5*maxHeight)+botTailPt.y-topTailPt.y;
			if (tailRect.height+tailRect.y>=targetFgImg.rows)
				tailRect.height = targetFgImg.rows-tailRect.y-1;
		}

		Mat tailImg = targetImg(tailRect);
		if (isTailAtRight)
			flip(tailImg, tailImg, 1);
		Mat lbpImg;
		localBinaryPatterns(tailImg, lbpImg, 1);
		//imshow("targetImg",targetImg);
		//imshow("tailImg",tailImg);
		//imshow("lbpImg",lbpImg);
		//waitKey(0);

		vector<float> tailHist;
		tailHist.resize(16, 0);
		float increment = 1.0 / lbpImg.rows / lbpImg.cols;
		for(int y = 0; y < lbpImg.rows; ++y){
			for(int x = 0; x < lbpImg.cols; ++x){
				int lbp = lbpImg.at<uchar>(y, x);
				int idx = lbp / 16;
				tailHist[idx] += increment;
			}
		}

		// write feature: head area and shape
		// needs a better method to define the "head boundary"


		double headArea = contourArea(headContour);
		features.at<float>(0,featCnt) = headArea/objArea;
		++featCnt;
		//****fout << headArea / obj.area << ',';

		if (useHeadFD) {
			vector<double> headFD;
			//headFD = featExtractor.getFourierDescriptor(headContour);
			headFD = featExtractor.getCorrelation(headContour, isTailAtRight);
			for(size_t k = 0; k < headFD.size(); ++k){
				features.at<float>(0,featCnt) = headFD[k];
				featCnt++;
			}
		}

		// write head lbp features
		Rect headRect = boundingRect(headContour);
		/*if (~isTailAtRight) {
			headRect.width = int(1.5*headRect.width);
			if (headRect.x+headRect.width>=targetFgImg.cols)
				headRect.width = targetFgImg.cols-headRect.x-1;
		}
		else {
			int prev_x = headRect.x;
			headRect.x = int(headRect.x-0.5*headRect.width);
			if (headRect.x<0)
				headRect.x = 0;
			headRect.width += prev_x-headRect.x;
		}*/
		Mat headImg = targetImg(headRect);
		//imshow("headImg",headImg);
		//waitKey(0);
		if (isTailAtRight)
			flip(headImg,headImg,1);
		Mat lbpImgHead;
		localBinaryPatterns(headImg, lbpImgHead, 1);
		//imshow("targetImg",targetImg);
		//imshow("headImg",headImg);
		//waitKey(0);

		vector<float> headHist;
		headHist.resize(16, 0);
		increment = 1.0 / lbpImgHead.rows / lbpImgHead.cols;
		for(int y = 0; y < lbpImgHead.rows; ++y){
			for(int x = 0; x < lbpImgHead.cols; ++x){
				int lbp = lbpImgHead.at<uchar>(y, x);
				int idx = lbp / 16;
				headHist[idx] += increment;
			}
		}

		// head hog features
		vector<float> hogFeat;
		Mat uniHeadImg;
		resize(headImg, uniHeadImg, Size(32,32), 0, 0, INTER_CUBIC);
		//imshow("uniHeadImg",uniHeadImg);
		//waitKey(0);
		getHOGFeat(uniHeadImg, hogFeat);

		for(size_t k = 0; k < 36; ++k){
			features.at<float>(0,featCnt) = hogFeat[k];
			featCnt++;
		}

		// middle region
		Rect middleRect = boundingRect(middleContour);
		Mat middleImg = targetImg(middleRect);
		if (isTailAtRight)
			flip(middleImg,middleImg,1);
		Mat lbpImgMiddle;
		localBinaryPatterns(middleImg, lbpImgMiddle, 1);
		//imshow("middleImg",middleImg);
		//waitKey(0);
		vector<float> middleHist;
		middleHist.resize(16, 0);
		increment = 1.0 / lbpImgMiddle.rows / lbpImgMiddle.cols;
		for(int y = 0; y < lbpImgMiddle.rows; ++y){
			for(int x = 0; x < lbpImgMiddle.cols; ++x){
				int lbp = lbpImgMiddle.at<uchar>(y, x);
				int idx = lbp / 16;
				middleHist[idx] += increment;
			}
		}



		// write feature: max vertical projection / min vertical projection
		double projHeightRatio = (double)maxSum / (double)minSum;
		features.at<float>(0,featCnt) = projHeightRatio;
		++featCnt;

		// write feature: aspect ratio
		double bw = norm(obj.uPoints[0] - obj.uPoints[1]);
		double bh = norm(obj.uPoints[0] - obj.uPoints[3]);
		double aspRatio = max(bw/bh, bh/bw);
		features.at<float>(0,featCnt) = double(maxHeight)/double(bodyLength);
		++featCnt;
		//******fout << aspRatio << ',';

		// write feature: standard length / total length
		Point mouthPoint;
		int tipX = isTailAtRight ? 10000 : 0;
		for(int u = 0; u < targetContour.size(); ++u){
			if(isTailAtRight && targetContour[u].x < tipX || !isTailAtRight && targetContour[u].x > tipX){
				mouthPoint = targetContour[u];
				tipX = targetContour[u].x;
			}
		}
		//circle(targetFgImgC3, mouthPoint, radius, Scalar(0, 128, 255), -1);
		features.at<float>(0,featCnt) = abs(mouthPoint.x - tailPointX) / fishLen;
		++featCnt;


		/////////// eye detection using cascade classifier
		//Rect roiRect;
		if(isTailAtRight){
			roiRect = Rect(mouthPoint.x, boundingBox.y, neckPointX - mouthPoint.x, max(int(boundingBox.height*0.6), mouthPoint.y - boundingBox.y));
		}
		else{
			roiRect = Rect(neckPointX, boundingBox.y, mouthPoint.x - neckPointX, max(int(boundingBox.height*0.6), mouthPoint.y - boundingBox.y));
		}

		roiRect = boundingRect(targetContour);//*******
		Mat roi = targetImg(roiRect);
		//equalizeHist(roi, roi);

		//CascadeClassifier eyeDetector("C:\\Users\\ipl333\\Documents\\Code\\codes\\Supervised SpeciesID\\eyes\\cascade\\cascade.xml");
		CascadeClassifier eyeDetector("cascade\\cascade.xml");
		vector<Rect> eyes;
		eyes.reserve(16);
		int minDiameter = int(0.1*boundingBox.height + 0.5);
		int maxDiameter = int(0.6*boundingBox.height + 0.5);
		eyeDetector.detectMultiScale(roi, eyes, 1.1, 3, 0, Size(minDiameter, minDiameter), Size(maxDiameter, maxDiameter));

		Rect finalEye;
		int binWidth = 16;
		int lenHist = 256 / binWidth;

		if(eyes.empty()){
			//for(int p = 0; p < 36; ++p)
			//	fout << '0' << ',';
			for(int p = 0; p < lenHist+1; ++p) {
				features.at<float>(0,featCnt) = 0;
				featCnt++;
			}
		}
		else{

			int argmax = -1;
			float maxDist = 0;

			Point imgCen (targetImg.cols/2, targetImg.rows/2);
			for(size_t k = 0; k < eyes.size(); ++k){
				Point eyeCen (roiRect.x + eyes[k].x + eyes[k].width/2, roiRect.y + eyes[k].y + eyes[k].height/2);

				// use only the eyes that are inside body region
				/*Mat largeFg;
				Mat element = getStructuringElement(MORPH_ELLIPSE,
                                     Size(int(maxHeight/5+0.5), int(maxHeight/5+0.5)),
                                     Point(-1,-1) );
				dilate(targetFgImg, largeFg, element);*/

				//imshow("largeFg",largeFg);
				//imshow("targetFgImg",targetFgImg);
				//waitKey(0);
				if (abs(eyeCen.x-headPt.x)>bodyLength/5.0 && abs(eyeCen.x-topTailPt.x)>bodyLength/5.0)
					if (eyes[k].width<0.9*maxHeight)
						continue;
				if(targetFgImg.at<uchar>(eyeCen) != 0){		//*****
					float dist = norm(eyeCen - imgCen);
					if(dist > maxDist){
						argmax = k;
						maxDist = dist;
					}
				}
			}

			if(argmax == -1){
				//for(int p = 0; p < 36; ++p)
				//	fout << '0' << ',';
				for(int p = 0; p < lenHist+1; ++p) {
					features.at<float>(0,featCnt) = 0;
					featCnt++;
				}
			}
			else{
				finalEye = eyes[argmax];
				Point finalEyeCen (roiRect.x + finalEye.x + finalEye.width/2, roiRect.y + finalEye.y + finalEye.height/2);
				circle(targetImgC3, finalEyeCen, finalEye.width/2, Scalar(0, 0, 255), finalEye.width/20);
				//imshow("targetImgC3",targetImgC3);

				finalEye.x = int(finalEyeCen.x-finalEye.width*1);
				if (finalEye.x<0)
					finalEye.x = 0;
				finalEye.y = int(finalEyeCen.y-finalEye.height*1);
				if (finalEye.y<0)
					finalEye.y = 0;
				finalEye.width = 2*finalEye.width;
				if (finalEye.x+finalEye.width>=targetImgC3.cols)
					finalEye.width = targetImgC3.cols-finalEye.x-1;
				finalEye.height = 2*finalEye.height;
				if (finalEye.y+finalEye.height>=targetImgC3.rows)
					finalEye.height = targetImgC3.rows-finalEye.y-1;
				Mat eyeImg = targetImg(finalEye);
				if (isTailAtRight)
					flip(eyeImg,eyeImg,1);
				//imshow("eyeImg",eyeImg);
				//waitKey(0);
				Mat lbpImg;
				localBinaryPatterns(eyeImg, lbpImg, 1);

				vector<float> hist;
				hist.resize(lenHist, 0);
				float increment = 1.0 / lbpImg.rows / lbpImg.cols;
				for(int y = 0; y < lbpImg.rows; ++y){
					for(int x = 0; x < lbpImg.cols; ++x){
						int lbp = lbpImg.at<uchar>(y, x);
						int idx = lbp / binWidth;
						hist[idx] += increment;
					}
				}

				// eye hog
				/*vector<float> hogFeatEye;
				Mat uniEyeImg;
				resize(eyeImg, uniEyeImg, Size(32,32), 0, 0, INTER_CUBIC);
				getHOGFeat(uniEyeImg, hogFeatEye);
				for(size_t k = 0; k < 36; ++k){
					fout << hogFeatEye[k] << ',';
				}*/


				double eyeArea = finalEye.width * finalEye.height * PI_ / 4.0;
				features.at<float>(0,featCnt) = eyeArea / headArea;
				++featCnt;
				for(int p = 0; p < lenHist; ++p){
					features.at<float>(0,featCnt) = hist[p];
					featCnt++;
				}

			}

		}

		///////////////////////
		for(int p = 0; p < 16; ++p){
			features.at<float>(0,featCnt) = tailHist[p];
			featCnt++;
		}

		for(int p = 0; p < 16; ++p){
			features.at<float>(0,featCnt) = headHist[p];
			featCnt++;
		}

		for(int p = 0; p < 16; ++p){
			features.at<float>(0,featCnt) = middleHist[p];
			featCnt++;
		}

		for (int p = 0; p < grad_hist.size(); p++) {
			features.at<float>(0,featCnt) = grad_hist[p];
			featCnt++;
		}
		//cout<<featCnt<<endl;
		//waitKey(0);

		// write taget image and segmentation
		//sprintf_s(imgPath, "%s\\%s\\fg-%03d.jpg", param.getResultPath(), param.getClassLabel(i), j + 1);
		//imwrite(imgPath, targetFgImgC3);
		//sprintf_s(imgPath, "%s\\eyes\\%s-%03d.jpg", param.getResultPath(), param.getClassLabel(i), j + 1);
		//imwrite(imgPath, targetImgC3);
		//cout<<endl;
	//}


	return 0;
}

