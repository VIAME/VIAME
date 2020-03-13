#include "featureExtraction.h"
#include "util.h"

void localBinaryPatterns(InputArray src, OutputArray dst, int radius)
{
	if(!src.obj) return;
	Mat img = src.getMat();
	dst.create(img.size(), CV_8U);
	Mat outImg = dst.getMat();

	for(int y = radius; y < img.rows - radius; ++y){
		for(int x = radius; x < img.cols - radius; ++x){
			Point cen(x, y);
			uchar cenVal = img.at<uchar>(cen);
			uchar lbpDecimal = 0;

			for(int t = 0; t < 8; ++t){
				float xx = cen.x + radius*cos(t*PI_/4.0);
				float yy = cen.y + radius*sin(t*PI_/4.0);

				float a = xx - floor(xx);
				float b = yy - floor(yy);

				uchar ptVal = uchar((1-a) * (1-b) * img.at<uchar>(int(floor(yy)), int(floor(xx)))
								  +     a * (1-b) * img.at<uchar>(int(floor(yy)), int(ceil(xx)))
								  + (1-a) *     b * img.at<uchar>(int(ceil(yy)), int(floor(xx)))
								  +     a *     b * img.at<uchar>(int(ceil(yy)), int(ceil(xx))) );

				if (ptVal > cenVal - 3){
					lbpDecimal += 1 << (7-t);
				}
			}

			outImg.at<uchar>(cen) = lbpDecimal;
			
		}
	}
}



vector<int>
FeatureExtraction::getCurvatureMaxIdx(const vector<Point>& contour)
{
	int nP = contour.size(); // number of sample points around the contour
	double s = 16.0; // Gaussian sigma

	vector<int> x, y;
	x.reserve(nP);
	y.reserve(nP);
	size_t size = contour.size();
	for(int u = 0; u < nP; ++u){
		x.push_back(contour[u*size/nP].x);
		y.push_back(contour[u*size/nP].y);
	}


	// 1D gaussian kernel
	vector<double> g, gu, guu;
	for(int v = 0; v <= 4*s; ++v){
		double G = exp(-0.5 * pow((v-2*s)/s, 2)) / sqrt(2*PI_) / s;
		g.push_back(G);
		gu.push_back(-(v-2*s)/pow(s, 2) * G);
		guu.push_back((-pow(s, 2) + pow(v-2*s, 2)) / pow(s, 4) * G);
	}

	vector<double> X, Xu, Xuu, Y, Yu, Yuu, k;
	X.reserve(nP);
	Xu.reserve(nP);
	Xuu.reserve(nP);
	Y.reserve(nP);
	Yu.reserve(nP);
	Yuu.reserve(nP);
	k.reserve(nP);

	double maxAbsK = 0;
	for(int i = 0; i < nP; ++i){
		X.push_back(0);
		Xu.push_back(0);
		Xuu.push_back(0);
		Y.push_back(0);
		Yu.push_back(0);
		Yuu.push_back(0);

		for(int j = 0; j <= 4*s; ++j){
			int idx = int(i-j+2*s);
			idx = idx < 0 ? idx + nP : (idx >= nP ? idx - nP : idx);
			X[i]   += x[idx] * g[j];
			Xu[i]  += x[idx] * gu[j];
			Xuu[i] += x[idx] * guu[j];
			Y[i]   += y[idx] * g[j];
			Yu[i]  += y[idx] * gu[j];
			Yuu[i] += y[idx] * guu[j];
		}
		double ki = (Xu[i]*Yuu[i] - Xuu[i]*Yu[i]) / pow((Xu[i]*Xu[i] + Yu[i]*Yu[i]), 1.5);
		k.push_back(ki);
		if(abs(ki) > maxAbsK) maxAbsK = abs(ki);
	}

	vector<int> curvatureMaxIdx;
	double prevK, nextK;
	for(int i = 0; i < k.size(); ++i){
		prevK = k[i-1 >= 0 ? i-1 : k.size()-1];
		nextK = k[i+1 < k.size() ? i+1 : 0];
		double absKi = abs(k[i]);
		if(absKi > 0.1 * maxAbsK && abs(prevK) < absKi && abs(nextK) < absKi){
			int idx = int(i * size / (float)nP + 0.5);
			curvatureMaxIdx.push_back(idx);
		}
	}

	return curvatureMaxIdx;
}

vector<double>
FeatureExtraction::getCurvature(const vector<Point>& contour)
{
	int nP = contour.size();
	double s = 4.0; // Gaussian sigma

	vector<int> x, y;
	x.reserve(nP);
	y.reserve(nP);
	size_t size = contour.size();
	for(int u = 0; u < nP; ++u){
		x.push_back(contour[u].x);
		y.push_back(contour[u].y);
	}


	// 1D gaussian kernel
	vector<double> g, gu, guu;
	for(int v = 0; v <= 4*s; ++v){
		double G = exp(-0.5 * pow((v-2*s)/s, 2)) / sqrt(2*PI_) / s;
		g.push_back(G);
		gu.push_back(-(v-2*s)/pow(s, 2) * G);
		guu.push_back((-pow(s, 2) + pow(v-2*s, 2)) / pow(s, 4) * G);
	}

	vector<double> X, Xu, Xuu, Y, Yu, Yuu, kappa;
	X.reserve(nP);
	Xu.reserve(nP);
	Xuu.reserve(nP);
	Y.reserve(nP);
	Yu.reserve(nP);
	Yuu.reserve(nP);
	kappa.reserve(nP);
	for(int i = 0; i < nP; ++i){
		X.push_back(0);
		Xu.push_back(0);
		Xuu.push_back(0);
		Y.push_back(0);
		Yu.push_back(0);
		Yuu.push_back(0);
		for(int j = 0; j <= 4*s; ++j){
			int idx = int(i-j+2*s);
			idx = idx < 0 ? idx + nP : (idx >= nP ? idx - nP : idx);
			X[i]   += x[idx] * g[j];
			Xu[i]  += x[idx] * gu[j];
			Xuu[i] += x[idx] * guu[j];
			Y[i]   += y[idx] * g[j];
			Yu[i]  += y[idx] * gu[j];
			Yuu[i] += y[idx] * guu[j];
		}
		double k = (Xu[i]*Yuu[i] - Xuu[i]*Yu[i]) / pow((Xu[i]*Xu[i] + Yu[i]*Yu[i]), 1.5);
		kappa.push_back(k);

	}

	return kappa;
}

vector<pair<double, int>>
FeatureExtraction::getCSSMaxima(const FGObject& obj, OutputArray cssImg)
{
	vector<Point> contour = obj.contour;
	int nS = 200;
	int nP = 200;
	cssImg.create(nS, nP, CV_8U);
	Mat css = cssImg.getMat();
	
	cssImage(contour, css);
	
	Mat cssImg8U = css.clone();

	Mat se = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	dilate(cssImg8U, cssImg8U, se);

	vector<vector<Point>> cssContours = extractContours(cssImg8U);
	
    vector<pair<double, int>> cssMax;
	for(int j = cssContours.size()-1; j >= 0; --j){
		Point pt = cssContours[j][0];
		double s = (199.0 - pt.y)*0.15 + 1.0;
		if(j == cssContours.size()-1){
			cssMax.push_back(make_pair(s, pt.x));
			continue;
		}

		if(cssMax.size() > 0 && s <= cssMax.front().first / 5.0) break;

		pair<double, int> back = cssMax.back();
		// calculate the midpoint of two branch peaks
		if(abs(back.first - s) < 0.15 && abs(back.second - pt.x) < 7){
			int midX = (pt.x + back.second)/2;
			cssMax.pop_back();
			cssMax.push_back(make_pair(s, midX));
		}
		else{
			cssMax.push_back(make_pair(s, pt.x));
		}

	}

	return cssMax;
}

vector<double>
FeatureExtraction::getCurvature(const FGObject& obj)
{
	int nP = 50; // number of sample points around the contour
	int kw = 4; // Gaussian kernel width
	double s = 2.0; // Gaussian sigma

	vector<Point> contour = obj.contour;
	vector<int> x, y;
	x.reserve(nP);
	y.reserve(nP);
	size_t size = contour.size();
	for(int u = 0; u < nP; ++u){
		x.push_back(contour[u*size/nP].x);
		y.push_back(contour[u*size/nP].y);
	}


	// 1D gaussian kernel
	vector<double> g, gu, guu;
	for(int v = 0; v <= 2*kw*s; ++v){
		double G = exp(-0.5 * pow((v-kw*s)/s, 2)) / sqrt(2*PI_) / s;
		g.push_back(G);
		gu.push_back(-(v-kw*s)/pow(s, 2) * G);
		guu.push_back((-pow(s, 2) + pow(v-kw*s, 2)) / pow(s, 4) * G);
	}

	vector<double> X, Xu, Xuu, Y, Yu, Yuu, k;
	X.reserve(nP);
	Xu.reserve(nP);
	Xuu.reserve(nP);
	Y.reserve(nP);
	Yu.reserve(nP);
	Yuu.reserve(nP);
	k.reserve(nP);
	for(int i = 0; i < nP; ++i){
		X.push_back(0);
		Xu.push_back(0);
		Xuu.push_back(0);
		Y.push_back(0);
		Yu.push_back(0);
		Yuu.push_back(0);
		for(int j = 0; j <= 2*kw*s; ++j){
			int idx = int(i-j+kw*s);
			idx = idx < 0 ? idx + nP : (idx >= nP ? idx - nP : idx);
			X[i]   += x[idx] * g[j];
			Xu[i]  += x[idx] * gu[j];
			Xuu[i] += x[idx] * guu[j];
			Y[i]   += y[idx] * g[j];
			Yu[i]  += y[idx] * gu[j];
			Yuu[i] += y[idx] * guu[j];
		}
		double ki = (Xu[i]*Yuu[i] - Xuu[i]*Yu[i]) / pow((Xu[i]*Xu[i] + Yu[i]*Yu[i]), 1.5);
		k.push_back(ki);

	}

	return k;
}

vector<int>
FeatureExtraction::getConcaveIndices(const FGObject& obj)
{
	int nP = 100; // number of sample points around the contour
	int kw = 4; // Gaussian kernel width
	double s = 2.0; // Gaussian sigma

	vector<Point> contour = obj.contour;
	vector<int> x, y;
	x.reserve(nP);
	y.reserve(nP);
	size_t size = contour.size();
	for(int u = 0; u < nP; ++u){
		x.push_back(contour[u*size/nP].x);
		y.push_back(contour[u*size/nP].y);
	}

	
	// 1D gaussian kernel
	vector<double> g, gu, guu;
	for(int v = 0; v <= 2*kw*s; ++v){
		double G = exp(-0.5 * pow((v-kw*s)/s, 2)) / sqrt(2*PI_) / s;
		g.push_back(G);
		gu.push_back(-(v-kw*s)/pow(s, 2) * G);
		guu.push_back((-pow(s, 2) + pow(v-kw*s, 2)) / pow(s, 4) * G);
	}

	vector<double> X, Xu, Xuu, Y, Yu, Yuu, k;
	X.reserve(nP);
	Xu.reserve(nP);
	Xuu.reserve(nP);
	Y.reserve(nP);
	Yu.reserve(nP);
	Yuu.reserve(nP);
	k.reserve(nP);
	
	double maxK = 0;
	for(int i = 0; i < nP; ++i){
		X.push_back(0);
		Xu.push_back(0);
		Xuu.push_back(0);
		Y.push_back(0);
		Yu.push_back(0);
		Yuu.push_back(0);
		
		for(int j = 0; j <= 2*kw*s; ++j){
			int idx = int(i-j+kw*s);
			idx = idx < 0 ? idx + nP : (idx >= nP ? idx - nP : idx);
			X[i]   += x[idx] * g[j];
			Xu[i]  += x[idx] * gu[j];
			Xuu[i] += x[idx] * guu[j];
			Y[i]   += y[idx] * g[j];
			Yu[i]  += y[idx] * gu[j];
			Yuu[i] += y[idx] * guu[j];
		}
		double ki = (Xu[i]*Yuu[i] - Xuu[i]*Yu[i]) / pow((Xu[i]*Xu[i] + Yu[i]*Yu[i]), 1.5);
		k.push_back(ki);
		if(ki > maxK) maxK = ki;
	}

	vector<int> concave;
	double prevK, nextK;
	for(int i = 0; i < k.size(); ++i){
		prevK = k[i-1 >= 0 ? i-1 : k.size()-1];
		nextK = k[i+1 < k.size() ? i+1 : 0];
		if(k[i] > 0.001*maxK && prevK < k[i] && nextK < k[i]){
			int idx = int(i * size / (float)nP + 0.5);
			concave.push_back(idx);
		}
	}

	return concave;
}



void
FeatureExtraction::cssImage(vector<Point> contour, OutputArray cssImg)
{
	int nP = 200;
	int nS = 200;
	double step = 0.15;
	int kw = 4;

	cssImg.create(nS, nP, CV_8U);
	Mat outImg8U = cssImg.getMat();
	outImg8U.setTo(Scalar(0));

	vector<int> x, y;
	x.reserve(nP);
	y.reserve(nP);
	size_t size = contour.size();
	for(int u = 0; u < nP; ++u){
		x.push_back(contour[u*size/nP].x);
		y.push_back(contour[u*size/nP].y);
	}

	int r = nS-1;
	for(double s = 1.0; s < 1.14+(nS-1)*step; s+=step){
		// 1D gaussian kernel
		vector<double> g, gu, guu;
		for(int v = 0; v <= 2*kw*s; ++v){
			double G = exp(-0.5 * pow((v-kw*s)/s, 2)) / sqrt(2*PI_) / s;
			g.push_back(G);
			gu.push_back(-(v-kw*s)/pow(s, 2) * G);
			guu.push_back((-pow(s, 2) + pow(v-kw*s, 2)) / pow(s, 4) * G);
		}

		// convolution and calculate curvature
		vector<double> X, Xu, Xuu, Y, Yu, Yuu, k;
		vector<bool> dyLarge;
		X.reserve(nP);
		Xu.reserve(nP);
		Xuu.reserve(nP);
		Y.reserve(nP);
		Yu.reserve(nP);
		Yuu.reserve(nP);
		k.reserve(nP);
		dyLarge.reserve(nP);
		for(int i = 0; i < nP; ++i){
			X.push_back(0);
			Xu.push_back(0);
			Xuu.push_back(0);
			Y.push_back(0);
			Yu.push_back(0);
			Yuu.push_back(0);
			dyLarge.push_back(true);
			for(int j = 0; j <= 2*kw*s; ++j){
				int idx = int(i-j+kw*s);
				idx = idx < 0 ? idx + nP : (idx >= nP ? idx - nP : idx);
				X[i]   += x[idx] * g[j];
				Xu[i]  += x[idx] * gu[j];
				Xuu[i] += x[idx] * guu[j];
				Y[i]   += y[idx] * g[j];
				Yu[i]  += y[idx] * gu[j];
				Yuu[i] += y[idx] * guu[j];
			}
			double ki = (Xu[i]*Yuu[i] - Xuu[i]*Yu[i]) / pow((Xu[i]*Xu[i] + Yu[i]*Yu[i]), 1.5);
			k.push_back(ki);
		}

		/*for(int u = 1; u < nP-1; ++u){
			if(abs(Y[u] - Y[u-1]) < 0.5 && abs(Y[u] - Y[u+1]) < 0.5){
				for(int v = -1; v <= 1; ++v){
					int idx = u+v < 0 ? 0 : (u+v >= nP ? nP-1 : u+v);
					dyLarge[idx] = false;
				}
			}
		}*/

		for(int u = 0; u < nP; ++u){
			if(k[u]*k[(u+1)%nP] < 0)
				outImg8U.at<uchar>(r, u) = 255;
			else
				outImg8U.at<uchar>(r, u) = 0;
		}
		r--;

		/*if(r % 10 == 9){
			vector<Point> cc;
			cc.reserve(200);
			for(int i = 0; i < nP; ++i)
				cc.push_back(Point(X[i], Y[i]));

			vector<vector<Point>> ccs;
			ccs.push_back(cc);
			Mat smoothContour = Mat::zeros(param.getFrameHeight(), param.getFrameWidth(), CV_8U);
			drawContours(smoothContour, ccs, 0, Scalar(255), -1);
			showImage("smoothContour", smoothContour, 1, 1);
			showImage("cssImg", outImg8U, 1);
			int a = 1;
		}*/
		
	}

	/*Mat se = getStructuringElement(MORPH_RECT, Size(1, 3));
	morphologyEx(outImg8U, outImg8U, CV_MOP_CLOSE, se);
	morphologyEx(outImg8U, outImg8U, CV_MOP_OPEN, se);*/

	/*vector<vector<Point> > cssContours = extractContours(outImg8U);
	for(size_t i = 0; i < cssContours.size(); ++i){
		Rect bRect = boundingRect(cssContours[i]);
		if(bRect.tl().y < 2 && bRect.height/(double)bRect.width > 4.0)
			drawContours(outImg8U, cssContours, i, Scalar(0), -1);
	}*/
	//showImage("CSS", outImg8U, 1, 0);
}

float
FeatureExtraction::cssMatchingCost(const vector<pair<double, int>>& cssMax, const vector<pair<double, int>>& cssMaxRef)
{
	float cost = abs(cssMax[0].first - cssMaxRef[0].first);
	int shift = cssMax[0].second - cssMaxRef[0].second;
	for(size_t j = 1; j < cssMax.size(); ++j){
		if(j >= cssMaxRef.size() || abs(cssMax[j].second - cssMaxRef[j].second - shift) > 0.2*100)
			cost += cssMax[j].first;
		else
			cost += abs(cssMax[j].first - cssMaxRef[j].first);
	}
	if(cssMax.size() < cssMaxRef.size()){
		for(size_t k = cssMax.size(); k < cssMaxRef.size(); ++k)
			cost += cssMaxRef[k].first;
	}

	return cost;
}

vector<double>
FeatureExtraction::getFourierDescriptor( const FGObject& obj )
{
	int nP = 100;
	
	vector<int> x, y;
	x.reserve(nP);
	y.reserve(nP);
	size_t size = obj.contour.size();
	for(int u = 0; u < nP; ++u){
		x.push_back(obj.contour[u*size/nP].x);
		y.push_back(obj.contour[u*size/nP].y);
	}

	vector<double> centDist;
	centDist.reserve(nP);
	for(int u = 0; u < nP; ++u){
		Point2f p (x[u], y[u]);
		double dist = norm(p - obj.uCenter);
		centDist.push_back(dist);
	}

	vector<complex<double>> DFT;
	DFT.reserve(centDist.size());
	DFT = discreteFourierTransform(centDist);

	vector<double> FD;
	FD.reserve(DFT.size());
	double dcMag = abs(DFT[0]);
	for(size_t i = 1; i < DFT.size()/2; ++i){
		double mag = abs(DFT[i]);
		FD.push_back(mag / dcMag);
	}

	return FD;
}

vector<double>
FeatureExtraction::getFourierDescriptor( const vector<Point>& contour )
{
	int nP = 40;

	vector<int> x, y;
	x.reserve(nP);
	y.reserve(nP);
	size_t size = contour.size();
	int sumX = 0, sumY = 0;
	for(int u = 0; u < nP; ++u){
		x.push_back(contour[u*size/nP].x);
		y.push_back(contour[u*size/nP].y);
		sumX += contour[u*size/nP].x;
		sumY += contour[u*size/nP].y;
	}

	Point2f cen (sumX / nP, sumY / nP);

	vector<double> centDist;
	centDist.reserve(nP);
	for(int u = 0; u < nP; ++u){
		Point2f p (x[u], y[u]);
		double dist = norm(p - cen);
		centDist.push_back(dist);
	}

	vector<complex<double>> DFT;
	DFT.reserve(centDist.size());
	DFT = discreteFourierTransform(centDist);

	vector<double> FD;
	FD.reserve(DFT.size()/2);
	double dcMag = abs(DFT[0]);
	for(size_t i = 1; i < DFT.size()/2; ++i){
		double mag = abs(DFT[i]);
		FD.push_back(mag / dcMag);
	}

	return FD;
}

vector<double> FeatureExtraction::getCorrelation(const vector<Point>& contour, bool isTailAtRight) {
	int nP = 40;

	vector<int> x, y;
	x.reserve(nP);
	y.reserve(nP);
	size_t size = contour.size();
	int sumX = 0, sumY = 0;
	int root = contour[0].x;
	for(int u = 0; u < nP; ++u){
		x.push_back(contour[u*size/nP].x);
		y.push_back(contour[u*size/nP].y);
		sumX += contour[u*size/nP].x;
		sumY += contour[u*size/nP].y;

		if (!isTailAtRight) {
			if (x[u]<root)
				root = x[u];
		}
		else {
			if (x[u]>root)
				root = x[u];
		}
	}

	Point2f cen (root, sumY / nP);

	vector<double> centDist;
	centDist.reserve(nP);
	for(int u = 0; u < nP; ++u){
		Point2f p (x[u], y[u]);
		double dist = norm(p - cen);
		centDist.push_back(dist);
	}

	// normalize
	double vectNorm = 0.0;
	for (int u = 0; u < nP; ++u) {
		vectNorm += centDist[u]*centDist[u];
	}
	vectNorm = sqrt(vectNorm);
	for (int u = 0; u < nP; ++u) {
		centDist[u] = centDist[u]/vectNorm;
	}

	vector<double> autoCorr(nP/2, 0.0);
	for (int n = 1; n <= autoCorr.size(); n++) {
		vector<double> x2 = centDist;
		rotate(x2.begin(),x2.begin()+n,x2.end());
		//for (int m = 0; m < n; m++) {
		//	x2[m] = 0.0;
		//}
		for (int u = 0; u < nP; u++)
			autoCorr[n-1] += centDist[u]*x2[u];
	}

	return autoCorr;
}


// returns the magnitude of the DFT of a real-number series
vector<complex<double>> FeatureExtraction::discreteFourierTransform( const vector<double>& inputSeries )
{
	int M = inputSeries.size();
	vector<complex<double>> DFT;
	DFT.reserve(inputSeries.size());
		
	for(int k = 0; k < M; ++k){
		double re = 0;
		double im = 0;
		for(int n = 0; n < M; ++n){
			double theta = -2 * PI_ * k * (double)n / (double)M;
			double cos_theta = cos(theta);
			double sin_theta = sin(theta);
			re += inputSeries[n] * cos_theta;
			im += inputSeries[n] * sin_theta;
		}
		
		complex<double> z (re, im);
		DFT.push_back(z);
	}

	return DFT;
}

/*void FeatureExtraction::getSIFTdescriptor( const FGObject& obj, InputArray src, OutputArray dst )
{
	if(!src.getObj()) return;
	Mat inImg = src.getMat();
	
	Size dsize;
	if(inImg.rows >= inImg.cols)
		dsize = Size(300, int(inImg.cols/(float)inImg.rows * 300 + 0.5));
	else
		dsize = Size(int(inImg.rows/(float)inImg.cols * 300 + 0.5), 300);
	Mat resizedImg;
	resize(inImg, resizedImg, dsize);


	// dense sampling for keypoints
	int patchWidth = 16;
	int step = 8;
	vector<KeyPoint> keypoints;
	
	
	for(int y = step; y < resizedImg.rows - step; y += step){
		for(int x = step; x < resizedImg.cols - step; x += step){
			float angle = principalOrientation(resizedImg, x, y, patchWidth);
			KeyPoint kp(x, y, patchWidth, angle);
			keypoints.push_back(kp);
		}
	}

	Mat descriptors;
	SiftDescriptorExtractor extractor;
	extractor.compute(resizedImg, keypoints, descriptors);

	dst.create(descriptors.size(), descriptors.type());
	Mat outMat = dst.getMat();
	descriptors.copyTo(outMat);
}*/

float
FeatureExtraction::principalOrientation(Mat img, int x, int y, int size)
{
	vector<int> histOri;
	histOri.reserve(8);
	for(int k = 0; k < 8; ++k)
		histOri.push_back(0);
	
	for(int j = y - size/2; j <= y + size/2; ++j){
		for(int i = x - size/2; i <= x + size/2; ++i){
			float diffX = 0, diffY = 0;
			if(j+1 >= img.rows)
				diffY = (float)img.at<uchar>(j, i) - (float)img.at<uchar>(j-1, i);
			else if(j-1 < 0)
				diffY = (float)img.at<uchar>(j+1, i) - (float)img.at<uchar>(j, i);
			else
				diffY = (float)img.at<uchar>(j+1, i) - (float)img.at<uchar>(j-1, i);
			
			if(i+1 >= img.cols)
				diffX = (float)img.at<uchar>(j, i) - (float)img.at<uchar>(j, i-1);
			else if(i-1 < 0)
				diffX = (float)img.at<uchar>(j, i+1) - (float)img.at<uchar>(j, i);
			else
				diffX = (float)img.at<uchar>(j, i+1) - (float)img.at<uchar>(j, i-1);
			
			int angle = int(atan2(diffY, diffX) / (0.25*PI_) + 4.0);
			if(angle == 8) angle = 7;
			++histOri[angle];
		}
	}
	

	int binHeight = 0;
	int idx = 0;
	for(int k = 0; k < 8; ++k){
		if(histOri[k] > binHeight){
			binHeight = histOri[k];
			idx = k;
		}
	}

	float ori = (idx - 4) * (0.25*PI_);

	return ori;
}

void getHOGFeat(Mat& src, vector<float>& dst) {
	int blockSize = 32;
	int blockStride = 16;
	int cellSize = 16;
	int nbin = 9;
	HOGDescriptor hog(src.size(), Size(blockSize, blockSize), Size(blockStride,blockStride), Size(cellSize,cellSize), nbin);
	vector<Point> locs;
	hog.compute(src, dst, Size(0,0), Size(0,0), locs);
}

vector<double> getGradHist(Mat img, Mat fg) {
	int ddepth = CV_16S;
	int scale = 1;
	int delta = 0;
	Mat grad;

	//GaussianBlur(img, img, Size(3,3), 0, 0, BORDER_DEFAULT);

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	Sobel(img, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	Sobel(img, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	double min, max;
	minMaxLoc(grad, &min, &max);

	vector<double> grad_hist(6,0.0);
	for (int m = 0; m < grad.rows; m++) {
		for (int n = 0; n < grad.cols; n++) {
			if (grad.at<uchar>(m,n)<16.0 || fg.at<uchar>(m,n)==0) 
				continue;
			if (grad.at<uchar>(m,n)>=16.0 && grad.at<uchar>(m,n)<60.0) {
				grad_hist[0]++;
				continue;
			}
			if (grad.at<uchar>(m,n)>=60.0 && grad.at<uchar>(m,n)<117.0) {
				grad_hist[1]++;
				continue;
			}
			if (grad.at<uchar>(m,n)>=117.0 && grad.at<uchar>(m,n)<150.0) {
				grad_hist[2]++;
				continue;
			}
			if (grad.at<uchar>(m,n)>=150.0 && grad.at<uchar>(m,n)<180.0) {
				grad_hist[3]++;
				continue;
			}
			if (grad.at<uchar>(m,n)>=180.0 && grad.at<uchar>(m,n)<235.0) {
				grad_hist[4]++;
				continue;
			}
			if (grad.at<uchar>(m,n)>=235.0) {
				grad_hist[5]++;
				continue;
			}
		}
	}

	double sum_v = grad_hist[0]+grad_hist[1]+grad_hist[2]+grad_hist[3]+grad_hist[4]+grad_hist[5];
	for (int n = 0; n < 6; n++)
		grad_hist[n] = grad_hist[n]/sum_v;
	//imshow("grad",grad);
	//waitKey(0);

	return grad_hist;
}