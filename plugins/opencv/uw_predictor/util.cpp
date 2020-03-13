#include <iostream>
using namespace std;
#include <cmath>
#include "util.h"
#include "FGObject.h"
#include "opencv2/highgui/highgui.hpp"

RNG rng(54321);

// wrapper of the routines to show an image in the window
void showImage(const string& winname, Mat img, int autosize, int delay)
{
	namedWindow(winname, autosize);
	imshow(winname, img);
	waitKey(delay);
}

// wrapper of the routines to find contours in a grayscale image
vector<vector<Point> > extractContours(const Mat& img)
{
	vector<vector<Point> > contours;
	Mat tempImg = img.clone();
	findContours(tempImg, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point());
	tempImg.release();
	return contours;
}

void plotOrientedBoundingBox(Mat& img, const RotatedRect& orientedBox, Scalar color)
{
	Point2f pts [4];
	orientedBox.points(pts);
	for(int i = 0; i < 4; ++i)
		line(img, pts[i], pts[(i+1)%4], color, 2);
}

// generate oriented bounding box
// find the rotation angle by principal component analysis (PCA)
RotatedRect orientedBoundingBox(const vector<Point>& contour)
{
	Mat mask8U = Mat::zeros(2048, 2048, CV_8U);	//****param.
	vector<vector<Point>> cs;
	cs.push_back(contour);
	drawContours(mask8U, cs, 0, Scalar(255), -1);
	//showImage("mask8U", mask8U, 1);

	Mat data = Mat::zeros(2, contour.size(), CV_32F);
	for(int j = 0; j < contour.size(); ++j){
		data.at<float>(0, j) = contour[j].x;
		data.at<float>(1, j) = contour[j].y;
	}

	RotatedRect orientedBox;
	if(contour.size() <= 2){
		if(contour.size() == 1){
			orientedBox.center = contour[0];
		}
		else{
			orientedBox.center.x = 0.5f*(contour[0].x + contour[1].x);
			orientedBox.center.y = 0.5f*(contour[0].x + contour[1].x);
			double dx = contour[1].x - contour[0].x;
			double dy = contour[1].y - contour[0].y;
			orientedBox.size.width = (float)sqrt(dx*dx + dy*dy);
			orientedBox.size.height = 0;
			orientedBox.angle = (float)atan2(dy, dx) * 180 / CV_PI;
		}
		return orientedBox;
	}

	PCA pcaObj = PCA(data, noArray(), CV_PCA_DATA_AS_COL);

	Mat result;
	pcaObj.project(data, result);

	// find two endpoints in principal component's direction
	float maxU = 0, maxV = 0;
	float minU = 0, minV = 0;

	for(int j = 0; j < result.cols; ++j){
		float u = result.at<float>(0, j);
		float v = result.at<float>(1, j);
		if(u > 0 && u > maxU)
			maxU = u;
		else if(u < 0 && u < minU)
			minU = u;

		if(v > 0 && v > maxV)
			maxV = v;
		else if(v < 0 && v < minV)
			minV = v;
	}

	float cenU = 0.5*(maxU + minU);
	float cenV = 0.5*(maxV + minV);

	Mat cenUVMat = (Mat_<float>(2, 1) << cenU, cenV);
	Mat cenXYMat = pcaObj.backProject(cenUVMat);

	Point cen(cenXYMat.at<float>(0, 0), cenXYMat.at<float>(1, 0));

	float width = maxU - minU;
	float height = maxV - minV;

	Mat pc = pcaObj.eigenvectors;
	float pcx = pc.at<float>(0, 0);
	float pcy = pc.at<float>(0, 1);
	float theta = atan2(pcy, pcx) * 180 / CV_PI;

	orientedBox.center = cen;
	orientedBox.size = Size2f(width, height);
	orientedBox.angle = theta;
	return orientedBox;
}


// non-maximal suppression on a 2-D image
//  *  Author:  Hilton Bristow
//	*  Created: Jul 19, 2012
void nonMaxSuppression(const Mat& src, const int sz, Mat& dst, const Mat mask)
{
	// initialize the block mask and destination
	const int M = src.rows;
	const int N = src.cols;
	const bool masked = !mask.empty();
	Mat block = 255*Mat_<uchar>::ones(Size(2*sz+1, 2*sz+1));
	dst = Mat_<uchar>::zeros(src.size());

	// iterate over image blocks
	for (int m = 0; m < M; m+=sz+1) {
		for (int n = 0; n < N; n+=sz+1) {
			Point  ijmax;
			double vcmax, vnmax;

			// get the maximal candidate within the block
			Range ic(m, min(m+sz+1,M));
			Range jc(n, min(n+sz+1,N));
			minMaxLoc(src(ic,jc), NULL, &vcmax, NULL, &ijmax, masked ? mask(ic,jc) : noArray());
			Point cc = ijmax + Point(jc.start,ic.start);

			// search the neighbors centered around the candidate for the true maxima
			Range in(max(cc.y-sz,0), min(cc.y+sz+1,M));
			Range jn(max(cc.x-sz,0), min(cc.x+sz+1,N));

			// mask out the block whose maxima we already know
			Mat_<uchar> blockmask;
			block(Range(0,in.size()), Range(0,jn.size())).copyTo(blockmask);
			Range iis(ic.start-in.start, min(ic.start-in.start+sz+1, in.size()));
			Range jis(jc.start-jn.start, min(jc.start-jn.start+sz+1, jn.size()));
			blockmask(iis, jis) = Mat_<uchar>::zeros(Size(jis.size(),iis.size()));

			minMaxLoc(src(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? mask(in,jn).mul(blockmask) : blockmask);
			Point cn = ijmax + Point(jn.start, in.start);

			// if the block center is also the neighbor center, then it's a local maxima
			if (vcmax > vnmax) {
				dst.at<uchar>(cc.y, cc.x) = 255;
			}
		}
	}
}

vector<vector<int>> combnk(vector<vector<int>> buff, vector<int> input, int k) {
	int buff_size = buff.size();
	int num = input.size();
	if (k==0) {
		vector<vector<int>> final_buff;
		for (int n = 0; n < buff.size(); n++)
			if (buff[n].back()!=-1)
				final_buff.push_back(buff[n]);
		return final_buff;
	}

	vector<vector<int>> total_buff;
	for (int n = 0; n < num; n++) {
		vector<int> temp_input = input;
		vector<vector<int>> temp_buff = buff;
		if (buff_size==0) {
			vector<int> tt;
			tt.push_back(temp_input[n]);
			temp_buff.push_back(tt);
			temp_input.erase(temp_input.begin()+n);
			vector<vector<int>> new_buff = combnk(temp_buff, temp_input, k-1);
			int new_num = new_buff.size();
			for (int kk = 0; kk < new_num; kk++)
				total_buff.push_back(new_buff[kk]);
		}
		for (int m = 0; m < buff_size; m++) {
			if (temp_buff[m].back()!=-1 && temp_buff[m].back()<temp_input[n])
				temp_buff[m].push_back(temp_input[n]);
			else
				temp_buff[m].push_back(-1);
			temp_input.erase(temp_input.begin()+n);
			vector<vector<int>> new_buff = combnk(temp_buff, temp_input, k-1);
			int new_num = new_buff.size();
			for (int kk = 0; kk < new_num; kk++)
				total_buff.push_back(new_buff[kk]);
		}
	}
	return total_buff;
}

// crop the target from input image and rotate to horizontal orientation
Rect outputTargetImage(const FGObject& obj, InputArray src, InputArray fgSrc, OutputArray dst, OutputArray dstFg, Mat& R, Point& shift)
{
	if(!src.obj || !fgSrc.obj) return Rect();
	Mat inImg = src.getMat();
	Mat fgImg = fgSrc.getMat();
	/*circle(fgImg, obj.uPoints[0], 5, Scalar(255), -1);
	circle(fgImg, obj.uPoints[1], 5, Scalar(255), -1);
	circle(fgImg, obj.uPoints[2], 5, Scalar(255), -1);
	circle(fgImg, obj.uPoints[3], 5, Scalar(255), -1);
	imshow("fgImg",fgImg);*/
	//waitKey(0);

	float x_min = inImg.cols, y_min = inImg.rows, x_max = 0, y_max = 0;
	for(int k = 0; k < 4; ++k){
		if(obj.uPoints[k].x < x_min) x_min = obj.uPoints[k].x;
		if(obj.uPoints[k].y < y_min) y_min = obj.uPoints[k].y;
		if(obj.uPoints[k].x > x_max) x_max = obj.uPoints[k].x;
		if(obj.uPoints[k].y > y_max) y_max = obj.uPoints[k].y;
	}
	Point2f tl (x_min, y_min);
	Point2f br (x_max, y_max);

	tl += 0.8*(tl - obj.uCenter);
	br += 0.8*(br + obj.uCenter);
	tl.x = tl.x < 0 ? 0 : (tl.x > inImg.cols-1 ? inImg.cols-1 : tl.x);
	tl.y = tl.y < 0 ? 0 : (tl.y > inImg.rows-1 ? inImg.rows-1 : tl.y);
	br.x = br.x < 0 ? 0 : (br.x > inImg.cols-1 ? inImg.cols-1 : br.x);
	br.y = br.x < 0 ? 0 : (br.y > inImg.rows-1 ? inImg.rows-1 : br.y);

	shift.x = tl.x;
	shift.y = tl.y;
	Rect roiRect (tl, br);
	Mat roiImg = inImg(roiRect);
	Mat roiFgImg = fgImg(roiRect);
	//showImage("roiImg", roiImg, 1, 1);
	//showImage("roiFgImg", roiFgImg, 1);
	//waitKey(0);


	// preserve only the blob with largest area
	vector<vector<Point>> contours = extractContours(roiFgImg);
	double maxA = 0;
	int argMaxA = 0;
	for(int n = 0; n < contours.size(); ++n){
		double area = contourArea(contours[n]);
		if(area > maxA) {
			maxA = area;
			argMaxA = n;
		}
	}
	//showImage("old fg", targetFgImg, 1, 1);

	Mat tempFg = Mat::zeros(roiFgImg.size(), roiFgImg.type());
	drawContours(tempFg, contours, argMaxA, Scalar(255), -1);
	roiFgImg = tempFg;

	//showImage("new fg", targetFgImg, 1, 0);


	// rotate target by angle of bounding box
	Point2f roiCenter = obj.uCenter - tl;
	double angle = obj.angle;
	//cout << "angle = " << angle << endl;
	if(angle >= 90){
		angle = obj.angle - 180;
	}

	R = getRotationMatrix2D(roiCenter, angle, 1.0);

    // determine bounding rectangle
    Rect bbox = RotatedRect(roiCenter, fgImg.size(), angle).boundingRect();
    // adjust transformation matrix
    R.at<double>(0,2) += bbox.width/2.0 - roiCenter.x;
    R.at<double>(1,2) += bbox.height/2.0 - roiCenter.y;

	Mat rotatedRoiImg, rotatedRoiFgImg;
	Size dsize = roiImg.size();
	dsize.height = int(dsize.height*2);
	dsize.width = int(dsize.width*2);
	warpAffine(roiImg, rotatedRoiImg, R, bbox.size());
	warpAffine(roiFgImg, rotatedRoiFgImg, R, bbox.size());

	rotatedRoiImg.copyTo(dst);
	rotatedRoiFgImg.copyTo(dstFg);
	//showImage("rotatedRoiImg", rotatedRoiImg, 1, 1);
	//showImage("rotatedRoiFgImg", rotatedRoiFgImg, 1);
	//waitKey(0);

	/*tl.x = roiCenter.x - 0.7 * obj.uWidth;
	tl.y = roiCenter.y - 0.9 * obj.uHeight;
	br.x = roiCenter.x + 0.7 * obj.uWidth;
	br.y = roiCenter.y + 0.9 * obj.uHeight;
	tl.x = tl.x < 0 ? 0 : (tl.x > rotatedRoiImg.cols-1 ? rotatedRoiImg.cols-1 : tl.x);
	tl.y = tl.y < 0 ? 0 : (tl.y > rotatedRoiImg.rows-1 ? rotatedRoiImg.rows-1 : tl.y);
	br.x = br.x < 0 ? 0 : (br.x > rotatedRoiImg.cols-1 ? rotatedRoiImg.cols-1 : br.x);
	br.y = br.x < 0 ? 0 : (br.y > rotatedRoiImg.rows-1 ? rotatedRoiImg.rows-1 : br.y);
	roiRect = Rect(tl, br);

	Mat croppedRoiImg = rotatedRoiImg(roiRect);
	Mat croppedRoiFgImg = rotatedRoiFgImg(roiRect);

	//showImage("croppedRoiImg", croppedRoiImg, 1, 1);
	//showImage("croppedRoiFgImg", croppedRoiFgImg, 1);


	dst.create(croppedRoiImg.size(), croppedRoiImg.type());
	Mat targetImg = dst.getMat();
	dstFg.create(croppedRoiFgImg.size(), croppedRoiFgImg.type());
	Mat targetFgImg = dstFg.getMat();

	croppedRoiImg.copyTo(targetImg);
	croppedRoiFgImg.copyTo(targetFgImg);*/

	return roiRect;
}

Point computeOrigCoord(Mat R, Point inputPt) {
	Mat invR;
	invertAffineTransform(R, invR);

	Mat pt = Mat::zeros(3,1,CV_64FC1);
	pt.at<double>(0,0) = inputPt.x;
	pt.at<double>(1,0) = inputPt.y;
	pt.at<double>(2,0) = 1.0;

	Mat outPt(2,1,CV_64FC1);
	outPt = invR*pt;
	Point outputPt;
	outputPt.x = outPt.at<double>(0,0);
	outputPt.y = outPt.at<double>(1,0);
	return outputPt;
}
