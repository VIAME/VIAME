//////////////////////////////////////////////////////////////////////////
//
//  classHierarchy.h
//  Date:   Jun/12/2014
//  Author: Meng-Che Chuang, University of Washington
//
//  This file defines the data structures of class tree and tree nodes
//  for the hierarchical partial classification algorithm.
//

#include <map>

#include "classHierarchy.h"

//********** class classHierarchyNode ********************************************

ClassHierarchyNode::ClassHierarchyNode(int id) :
	_ID(id), _posClass(-1), _negClass(-1),
	_margin(0), _decThresh(0), _sigA(0), _sigB(0), _mu1(0), _mu2(0), _sigma1(1), _sigma2(1), _w1(0.5), _w2(0.5)
{
  _svm = std::shared_ptr< SVM >( new SVM );

	_svmParams.svm_type    = CvSVM::C_SVC;
	_svmParams.kernel_type = CvSVM::RBF;
	_svmParams.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 3000, 1e-4);
	//_svmParams.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-4);
}

ClassHierarchyNode::ClassHierarchyNode() :
	_ID(-1), _posClass(-1), _negClass(-1),
	_margin(0), _decThresh(0), _sigA(0), _sigB(0), _mu1(0), _mu2(0), _sigma1(1), _sigma2(1), _w1(0.5), _w2(0.5)
{
  _svm = std::shared_ptr< SVM >( new SVM );
}

ClassHierarchyNode::~ClassHierarchyNode()
{

}

void ClassHierarchyNode::trainSVM(Mat trainData, Mat labels)
{
	if(trainData.rows != labels.rows){
		cerr << "Error: number of training data and labels are not equal." << endl;
		return;
	}


	int k_fold = 10;

	// count numbers of positive and negative data
	int posCount = 0;
	int negCount = 0;
	for(int i = 0; i < trainData.rows; ++i){
		int label = (int)(labels.at<float>(i, 0));
		if(label == -1)
			++negCount;
		else if(label == 1)
			++posCount;
		else{
			cerr << "Error: more than two classes in training data." << endl;
			return;
		}
	}

	// set weights to handle class imbalance
	float count = posCount + negCount;
	Mat weights = (Mat_<float>(1, 2) << negCount / count, posCount / count);
	CvMat cvWeights = weights;
	_svmParams.class_weights = &cvWeights;

	// train the SVM with cross-validation
	_svm->train_auto(trainData, labels, Mat(), Mat(), _svmParams, k_fold);
	//_svm->train(trainData, labels, Mat(), Mat(), _svmParams);

	// TODO: Not sure what this is for. Remove this?
	_margin = decFuncMargin();

	//********************************************************************************

	// train the indecision threshold used by partial classification
	_decThresh = trainPartial(trainData, labels);	//*****

	// fit a sigmoid function for probabilistic output
	Mat decVal(labels.size(), CV_32F);
	for(int i = 0; i < decVal.rows; ++i){
		Mat instance = trainData.row(i);
		float f = _svm->predict(instance, true);
		decVal.at<float>(i, 0) = f;
	}

	pair<double, double> sigCoeff = fitSigmoid(decVal, labels, posCount, negCount);

	_sigA = sigCoeff.first;
	_sigB = sigCoeff.second;

	//******
	vector<float> param_gauss = fitProb(decVal, labels, posCount, negCount);
	_mu1 = param_gauss[0];
	_mu2 = param_gauss[1];
	_sigma1 = param_gauss[2];
	_sigma2 = param_gauss[3];
	_w1 = param_gauss[4];
	_w2 = param_gauss[5];
}

// calculate the decision function margin from all support vectors
double ClassHierarchyNode::decFuncMargin()
{
	int svCount = _svm->get_support_vector_count();
	int dim = _svm->get_var_count();

	double s = 0;
	for(int j = 0; j < svCount; ++j){
		const float* sv = _svm->get_support_vector(j);
		float temp = 0;
		for(int k = 0; k < dim; ++k){
			temp += sv[k] * sv[k];
		}
		s += sqrt(temp);
	}
	return (s / (double)svCount);
}

// probabilistic output for SVM by fitting a sigmoid function
// implemented based on the pseudo-code in [Platt 1999]
//     trainData  - training data
//     labels     - training data labels (1 or -1)
//     posCount   - number of positive samples
//     negCount   - number of negative samples
pair<double, double> ClassHierarchyNode::fitSigmoid_old(Mat decVals, Mat labels, int posCount, int negCount)
{
	double A = 0;
	double B = log((negCount + 1) / (double)(posCount + 1));
	double hiTarget = (posCount + 1) / (double)(posCount + 2);
	double loTarget = 1.0 / (double)(negCount + 2);
	double lambda = 1e-3;
	double oldErr = 1e10;

	vector<double> pp; // probabilities (?)
	pp.resize(posCount + negCount, (double)(posCount + 1) / (negCount + posCount + 2));
	int count = 0;

	// iterate to fit a sigmoid function to data
	int dataCount = posCount + negCount;
	for(size_t k = 0; k < 100; ++k){
		double a, b, c, d, e;
		a = b = c = d = e = 0;

		for(size_t i = 0; i < dataCount; ++i){
			int label = (int)labels.at<float>(i, 0);
			double t = label == 1 ? hiTarget : loTarget;
			double d1 = pp[i] - t;
			double d2 = pp[i]*(1 - pp[i]);

			a += pow(decVals.at<float>(i, 0), 2) * d2;
			b += d2;
			c += decVals.at<float>(i, 0) * d2;
			d += decVals.at<float>(i, 0) * d1;
			e += d1;
		}

		// if gradient is really tiny, then stop
		if(abs(d) < 1e-9 && abs(e) < 1e-9)
			break;

		double oldA = A;
		double oldB = B;
		double err = 0;

		// loop until goodness of fit increases
		while(true){
			double det = (a + lambda)*(b + lambda) - c*c;
			if(det == 0){ // if determinant of Hessian is zero,
				// increase stabilizer
				lambda *= 10;
				continue;
			}

			A = oldA + ((b + lambda)*d - c*e) / det;
			B = oldB + ((a + lambda)*e - c*d) / det;

			// now, compute the goodness of fit
			err = 0;
			for(size_t i = 0; i < negCount+posCount; ++i){
				int label = (int)labels.at<float>(i, 0);
				double t = label == 1 ? hiTarget : loTarget;
				double p = 1.0 / (1.0 + exp(A * decVals.at<float>(i, 0) + B));
				pp[i] = p;

				// at this step, make sure log(0) returns -200
				err -= t*std::max(-200.0, log(p)) + (1-t)*std::max(-200.0, log(1-p));
			}
			if(err < oldErr*(1 + 1e-7)){
				lambda *= 0.1;
				break;
			}

			// error did not decrease : increase stabilizer
			lambda *= 10;
			if(lambda >= 1e6) // something is broken
				break;
		}

		double diff = err - oldErr;
		double scale = 0.5*(err + oldErr + 1);
		if(diff > -1e-3*scale && diff < 1e-7*scale)
			++count;
		else
			count = 0;
		oldErr = err;

		if(count == 3) break;
	}

	return make_pair(A, B);
}

// probabilistic output for SVM by fitting a sigmoid function
// implemented based on the improved pseudo-code in [Lin 2007]
//     trainData  - training data
//     labels     - training data labels (1 or -1)
//     posCount   - number of positive samples
//     negCount   - number of negative samples
pair<double, double> ClassHierarchyNode::fitSigmoid(Mat decVals, Mat labels, int posCount, int negCount)
{
	// Parameter setting
	int maxiter = 100;	// Maximum number of iterations
	double minstep = 1e-10; // Minimum step taken in line search
	double sigma = 1e-12;	// Set to any value > 0

	// Construct initial values: target support in array t,
	// initial function value in fval
	double hiTarget = (posCount + 1.0) / (posCount + 2.0);
	double loTarget = 1 / (negCount + 2.0);

	double count = posCount + negCount; // Total number of data
	vector<double> t(count, 0);
	for (int i = 0; i < count; ++i) {
		int label = (int)labels.at<float>(i, 0);
		t[i] = label > 0 ? hiTarget : loTarget;
	}

	double A = 0.0;
	double B = log((negCount + 1.0) / (posCount + 1.0));
	double fval = 0.0;
	for (int i = 0; i < count; ++i) {
		double fApB = decVals.at<float>(i, 0)*A + B;
		if (fApB >= 0)
			fval += t[i]*fApB + log(1+exp(-fApB));
		else
			fval += (t[i]-1)*fApB + log(1+exp(fApB));
	}

	int k;
	for (k = 0; k < maxiter; ++k) {
		// Update Gradient and Hessian (use H¡¦ = H + sigma I)
		double h11, h22, h21, g1, g2;
		h11 = h22 = sigma;
		h21 = g1 = g2 = 0.0;
		for (int i = 0; i < count; ++i) {
			double f = decVals.at<float>(i, 0);
			double fApB = f*A + B;
			double p = fApB >= 0 ? exp(-fApB)/(1.0 + exp(-fApB)) : 1.0/(1.0 + exp(fApB));
			double q = fApB >= 0 ? 1.0/(1.0 + exp(-fApB)) : exp(fApB)/(1.0 + exp(fApB));

			double d2 = p*q;
			h11 += f * f * d2;
			h22 += d2;
			h21 += f * d2;

			double d1 = t[i] - p;
			g1 += f * d1;
			g2 += d1;
		}
		if (abs(g1) < 1e-5 && abs(g2) < 1e-5) // Stopping criteria
			break;

		// Compute modified Newton directions
		double det = h11*h22 - h21*h21;
		double dA = -(h22*g1 - h21*g2)/det;
		double dB = -(-h21*g1 + h11*g2)/det;
		double gd = g1*dA + g2*dB;

		// Line search
		double step = 1;
		while (step >= minstep){
			double newA = A + step*dA;
			double newB = B + step*dB;
			double newf = 0.0;

			for (int i = 0; i < count; ++i) {
				double fApB = decVals.at<float>(i, 0)*newA + newB;
				if (fApB >= 0)
					newf += t[i]*fApB + log(1 + exp(-fApB));
				else
					newf += (t[i] - 1)*fApB + log(1 + exp(fApB));
			}

			if (newf < fval + 0.0001*step*gd){
				A = newA;
				B = newB;
				fval = newf;
				break; // Sufficient decrease satisfied
			}
			else
				step /= 2.0;
		}

		if (step < minstep){
			cout << "figSigmoid: Line search fails." << endl;
			break;
		}
	}

	if (k >= maxiter)
		cout << "fitSigmoid: Reaching maximum iterations." << endl;

	return make_pair(A, B);
}

vector<float> ClassHierarchyNode::fitProb(Mat decVals, Mat labels, int posCount, int negCount) {
	// separate two class
	vector<float> posClass, negClass;
	float sum_pos = 0.0, sum_neg = 0.0;
	int cnt_pos = 0, cnt_neg = 0;
	float avg_pos, avg_neg;
	for (int n = 0; n < labels.rows; n++) {
		if (labels.at<float>(n,0)>0) {
			posClass.push_back(decVals.at<float>(n,0));
			cnt_pos++;
			sum_pos += decVals.at<float>(n,0);
		}
		else {
			negClass.push_back(decVals.at<float>(n,0));
			cnt_neg++;
			sum_neg += decVals.at<float>(n,0);
		}
	}
	avg_pos = sum_pos/cnt_pos;
	avg_neg = sum_neg/cnt_neg;

	// compute sigma
	float sigma_pos = 0.0, sigma_neg = 0.0;
	for (int n = 0; n < cnt_pos; n++) {
		sigma_pos += (posClass[n]-avg_pos)*(posClass[n]-avg_pos);
	}
	sigma_pos = sqrt(sigma_pos/cnt_pos);

	for (int n = 0; n < cnt_neg; n++) {
		sigma_neg += (negClass[n]-avg_neg)*(negClass[n]-avg_neg);
	}
	sigma_neg = sqrt(sigma_neg/cnt_neg);

	float w_pos = sqrt((float)cnt_pos)/(sqrt((float)cnt_pos)+sqrt((float)cnt_neg));
	float w_neg = 1-w_pos;
	//float w_neg = (float)cnt_neg/(float)labels.rows;

	vector<float> param_gauss(6);
	param_gauss[0] = avg_pos;
	param_gauss[1] = avg_neg;
	param_gauss[2] = sigma_pos;
	param_gauss[3] = sigma_neg;
	param_gauss[4] = w_pos;
	param_gauss[5] = w_neg;
	return param_gauss;
}

// train the decision threshold used in partial classification
// old approach: optimized by grid search
double ClassHierarchyNode::trainPartial(Mat decVals, Mat labels)
{
	printf("Train the indecision region of SVM #%d\n", _ID);

	int dataCount = decVals.rows;
	double lambda = 0.5;

	// store the minimum and maximum absolute decision values
	float fMin = 1e3;
	float fMax = 0;
	for(int i = 0; i < dataCount; ++i){
		float f = decVals.at<float>(i, 0);
		if(abs(f) < fMin) fMin = abs(f);
		if(abs(f) > fMax) fMax = abs(f);
	}

	//ofstream fout ("D:\\species_classification\\expBenefits_0209.txt", ios::app);

	float eta = (fMax - fMin) * 0.01 < 1e-4 ? 1e-4 : (fMax - fMin) * 0.01;
	int tpFull = -1;
	float maxBenefit = -1e6;

	double thresh = 0;

	for(float fID = fMin; fID <= fMax; fID += eta){
		int tpCount = 0; // true positives
		int fpCount = 0; // false positives
		int idCount = 0; // indecisions

		double sum1 = 0;
		double sum2 = 0;

		for(int i = 0; i < dataCount; ++i){
			float f = decVals.at<float>(i, 0);
			int label = (int)(labels.at<float>(i, 0) + 0.5);
			if(abs(f) < fID)
				++idCount;
			else{
				if(f > 0 && label == 1 || f < 0 && label == -1)
					++tpCount;
				else
					++fpCount;
			}

			int y = label == 1 ? 1 : -1;

			sum1 += exp(-y*f);
			sum2 += exp(-2*y*f);
		}

		if(tpFull < 0) tpFull = tpCount;

		double expBenefit = sum1/dataCount - exp(fID)*sum2/dataCount - exp(-fID) - lambda*(fID*fID);

		if(expBenefit > maxBenefit){
			thresh = fID;
			maxBenefit = expBenefit;
		}
	}

	return thresh;
}

// train the decision threshold used in partial classification
// optimized by applying the barrier method
double ClassHierarchyNode::trainPartialNew(Mat decVals, Mat labels)
{
	printf("Train the indecision region of SVM #%d\n", _ID);

	int dataCount = decVals.rows;
	double lambda = 0.5;

	// store the minimum and maximum absolute decision values
	float fMin = 1e3;
	float fMax = 0;
	for(int i = 0; i < dataCount; ++i){
		float f = decVals.at<float>(i, 0);
		if(abs(f) < fMin) fMin = abs(f);
		if(abs(f) > fMax) fMax = abs(f);
	}

	//ofstream fout ("D:\\species_classification\\expBenefits_0209.txt", ios::app);

	float eta = (fMax - fMin) * 0.01 < 1e-4 ? 1e-4 : (fMax - fMin) * 0.01;
	int tpFull = -1;
	float maxBenefit = -1e6;

	double thresh = 0;

	for(float fID = fMin; fID <= fMax; fID += eta){
		int tpCount = 0; // true positives
		int fpCount = 0; // false positives
		int idCount = 0; // indecisions

		double sum1 = 0;
		double sum2 = 0;

		for(int i = 0; i < dataCount; ++i){
			float f = decVals.at<float>(i, 0);
			int label = (int)(labels.at<float>(i, 0) + 0.5);
			if(abs(f) < fID)
				++idCount;
			else{
				if(f > 0 && label == 1 || f < 0 && label == -1)
					++tpCount;
				else
					++fpCount;
			}

			int y = label == 1 ? 1 : -1;

			sum1 += exp(-y*f);
			sum2 += exp(-2*y*f);
		}

		if(tpFull < 0) tpFull = tpCount;

		double expBenefit = sum1/dataCount - exp(fID)*sum2/dataCount - exp(-fID) - lambda*(fID*fID);

		if(expBenefit > maxBenefit){
			thresh = fID;
			maxBenefit = expBenefit;
		}
	}

	return thresh;
}

int ClassHierarchyNode::predictSVM( Mat sample, double& prob )
{
	int result = (int)_svm->predict(sample);
	double decVal = _svm->predict(sample, true);

	// get probabilistic output
	prob = 1.0 / (1.0 + exp(_sigA * decVal + _sigB));

	//prob = decVal;

	// label 0 if the sample is ambiguous
	if(abs(decVal) < _decThresh)
		result = 0;

	return result;
}

int ClassHierarchyNode::predictSVM2( Mat sample, double& prob_pos, double& prob_neg )
{
	int result = (int)_svm->predict(sample);
	double decVal = _svm->predict(sample, true);

	prob_pos = 1/(sqrt(2*PI_)*_sigma1)*exp(-(decVal-_mu1)*(decVal-_mu1)/(2*_sigma1*_sigma1))*_w1;
	prob_neg = 1/(sqrt(2*PI_)*_sigma2)*exp(-(decVal-_mu2)*(decVal-_mu2)/(2*_sigma2*_sigma2))*_w2;

	/*cout<<_sigma1<<endl;
	cout<<_sigma2<<endl;
	cout<<_mu1<<endl;
	cout<<_mu2<<endl;
	cout<<_w1<<endl;
	cout<<_w2<<endl;*/

	prob_pos = prob_pos/(prob_pos+prob_neg);
	prob_neg = 1-prob_pos;

	//cout<<prob_pos<<endl;
	//cout<<prob_neg<<endl;
	// label 0 if the sample is ambiguous
	if(abs(decVal) < _decThresh)
		result = 0;

	return result;
}

void ClassHierarchyNode::write(FileStorage& fs) const
{
	fs << "{";

	fs << "ID" << _ID;

	fs << "pos_class" << _posClass;
	fs << "neg_class" << _negClass;

	char file [32];
	snprintf(file, 200, "speciesSVM_%d.xml", _ID);
	_svm->save(file);

	fs << "svm_file" << file;
	fs << "margin" << _margin;
	fs << "dec_thresh" << _decThresh;
	fs << "sigmaA" << _sigA;
	fs << "sigmaB" << _sigB;

	fs << "mu1" << _mu1;
	fs << "mu2" << _mu2;
	fs << "sigma1" << _sigma1;
	fs << "sigma2" << _sigma2;
	fs << "w1" << _w1;
	fs << "w2" << _w2;

	fs << "}";
}

void ClassHierarchyNode::read(const FileNode& fn)
{
	_ID = (int)fn["ID"];

	_posClass = (int)fn["pos_class"];
	_negClass = (int)fn["neg_class"];

	string file = (string)fn["svm_file"];
	_svm->load(file.c_str());

	_margin = (double)fn["margin"];
	_decThresh = (double)fn["dec_thresh"];
	_sigA = (double)fn["sigmaA"];
	_sigB = (double)fn["sigmaB"];

	_mu1 = (double)fn["mu1"];
	_mu2 = (double)fn["mu2"];
	_sigma1 = (double)fn["sigma1"];
	_sigma2 = (double)fn["sigma2"];
	_w1 = (double)fn["w1"];
	_w2 = (double)fn["w2"];
}

void write(FileStorage& fs, const string& , const ClassHierarchyNode& x)
{
	x.write(fs);
}

void read(const FileNode& fn, ClassHierarchyNode& x, const ClassHierarchyNode& default_val)
{
	if(fn.empty())
		x = default_val;
	else
		x.read(fn);
}


//********** class classHierarchy ************************************************

ClassHierarchy::ClassHierarchy()
{
	_hierarchy.reserve(64);
}

ClassHierarchy::~ClassHierarchy()
{

}

void ClassHierarchy::train(Mat trainData, Mat trainLabels)
{
	if(trainData.rows != trainLabels.rows){
		cerr << "Error: number of training data and labels are not equal." << endl;
		return;
	}

	// count the number of species and number of data in each species
	vector<int> spcCounts;

	int dataCount = trainData.rows;
	for(int i = 0; i < dataCount; ++i){
		int label = (int)(trainLabels.at<float>(i, 0) + 0.5);
		if(label < 0){
			cerr << "Error: invalid class label (negative number) for data #" << i << endl;
			return;
		}
		if(label < spcCounts.size())
			++spcCounts[label];
		else
			spcCounts.push_back(1);
	}

	_nClasses = spcCounts.size();		// number of species

	// separate training data by species
	_speciesData.reserve(spcCounts.size());
	_speciesLabels.reserve(spcCounts.size());

	int idx = 0;
	for(int i = 0; i < spcCounts.size(); ++i){
		int n = spcCounts[i];
		Mat spcData = trainData.rowRange(idx, idx + n);
		_speciesData.push_back(spcData);
		Mat spcLabels = trainLabels.rowRange(idx, idx + n);
		_speciesLabels.push_back(spcLabels);
		idx += n;
	}

	// construct the class hierarchy by EM algorithm
	clusterClassesRecursive(trainData, trainLabels, 1);

}

bool ClassHierarchy::predict(Mat sample, vector<int>& predictions, vector<double>& probabilities)
{
	predictions.clear();
	probabilities.clear();

	bool isPartial = false;
	int i = 1;

	while(i < _hierarchy.size()){
		double prob = 0;
		int pred = _hierarchy[i].predictSVM(sample, prob);
		if(pred == 0){
			isPartial = true;
			predictions.push_back(-1);
			probabilities.push_back(prob);
			break;
		}

		int j = 2*i + (pred > 0 ? 0 : 1);
		if(j >= _hierarchy.size() || _hierarchy[j].getID() == -1){	//*****
			int sp = pred > 0 ? _hierarchy[i].getPosClass() : _hierarchy[i].getNegClass();
			predictions.push_back(sp);
			probabilities.push_back(prob);
			break;
		}

		predictions.push_back(j);
		probabilities.push_back(prob);
		i = j;
	}

	return isPartial;
}

bool ClassHierarchy::predict2(Mat sample, vector<int>& class_label, vector<vector<double>>& probb)
{
	//predictions.push_back(1);
	//probabilities.push_back(1.0);

	class_label.clear();
	probb.clear();

	vector<vector<int>> nodeID;
	//vector<vector<double>> probb;
	vector<int> canSplit;
	//vector<int> class_label;
	bool isPartial = false;
	int i = 1;

	vector<int> temp_node(1);
	temp_node[0] = 1;
	nodeID.push_back(temp_node);
	vector<double> temp_prob(1);
	temp_prob[0] = 1.0;
	probb.push_back(temp_prob);
	canSplit.push_back(1);
	class_label.push_back(-1);
	int min_node = 1;
	int cumsplit = 1;

	while(min_node < _hierarchy.size() && !isPartial && cumsplit!=0){
		for (int n = 0; n < nodeID.size(); n++) {
			cumsplit = 0;
			if (canSplit[n]==0) {
				cumsplit += canSplit[n];
				continue;
			}
			int current_node = nodeID[n].back();
			double prob = 0, prob_pos = 0.0, prob_neg = 0.0;
			int pred = _hierarchy[current_node].predictSVM2(sample, prob_pos, prob_neg);
			if(pred == 0){
				isPartial = true;
				break;
			}

			int node1 = 2*current_node;
			int node2 = 2*current_node+1;

			vector<int> temp_node1 = nodeID[n];
			temp_node1.push_back(node1);
			vector<int> temp_node2 = nodeID[n];
			temp_node2.push_back(node2);
			nodeID.push_back(temp_node1);
			nodeID.push_back(temp_node2);
			nodeID.erase(nodeID.begin()+n);

			vector<double> temp_prob1 = probb[n];
			temp_prob1.push_back(prob_pos);
			vector<double> temp_prob2 = probb[n];
			temp_prob2.push_back(prob_neg);
			probb.push_back(temp_prob1);
			probb.push_back(temp_prob2);
			probb.erase(probb.begin()+n);

			if(node1 >= _hierarchy.size() || _hierarchy[node1].getID() == -1) {
				canSplit.push_back(0);
				int sp = _hierarchy[current_node].getPosClass();
				class_label.push_back(sp);
			}
			else {
				canSplit.push_back(1);
				class_label.push_back(-1);
			}

			if(node2 >= _hierarchy.size() || _hierarchy[node2].getID() == -1) {
				canSplit.push_back(0);
				int sp = _hierarchy[current_node].getNegClass();
				class_label.push_back(sp);
			}
			else {
				canSplit.push_back(1);
				class_label.push_back(-1);
			}

			canSplit.erase(canSplit.begin()+n);
			class_label.erase(class_label.begin()+n);
		}

		min_node = nodeID[0].back();
		for (int n = 0; n < nodeID.size(); n++) {
			if (nodeID[n].back()<min_node)
				min_node = nodeID[n].back();
		}
	}

	return isPartial;
}

// recursively cluster the classes by using the EM algorithm
void ClassHierarchy::clusterClassesRecursive(Mat trainData, Mat trainLabels, int id)
{
	map<int, int> classCountMap;
	map<int, int>::iterator it;
	for(size_t i = 0; i < trainLabels.rows; ++i){
		int label = (int)(trainLabels.at<float>(i, 0) + 0.5);
		it = classCountMap.find(label);
		if(it == classCountMap.end())
			classCountMap[label] = 1;
		else
			++(it->second);
	}

	map<int, int>::iterator lastIt = classCountMap.end();
	--lastIt;
	cout << endl << "id = " << id << " : ";
	for(it = classCountMap.begin(); it != classCountMap.end(); ++it){
		cout << it->first << "(" << it->second << ")";
		if (it != lastIt)
			cout << ", ";
	}
	cout << endl;

	// base case : only one classes in these data
	if(classCountMap.size() == 1){
		it = classCountMap.begin();
		if(id % 2 == 0)
			_hierarchy[id/2].setPosClass(it->first);
		else
			_hierarchy[id/2].setNegClass(it->first);

		return;
	}

	// base case : only two classes in these data
	if(classCountMap.size() == 2){
		Mat svmLabels;
		for(size_t i = 0; i < trainData.rows; ++i){
			Mat data = trainData.row(i);
			int label = (int)(trainLabels.at<float>(i, 0) + 0.5);
			if(label == classCountMap.begin()->first)
				svmLabels.push_back(1.0f);
			else
				svmLabels.push_back(-1.0f);
		}

		if(_hierarchy.size() <= id)
			_hierarchy.resize(id + 1, ClassHierarchyNode(-1));
		_hierarchy[id].setID(id);
		_hierarchy[id].trainSVM(trainData, svmLabels);

		//double prob = 0;
		//Mat sample = trainData.row(0);
		//int pred = _hierarchy[id].predictSVM(sample, prob);

		it = classCountMap.begin();
		_hierarchy[id].setPosClass(it->first);
		_hierarchy[id].setNegClass((++it)->first);

		return;
	}

	//////////////////////////////////////////////////////////////////////////

	// compute gaussian model
	int nClass = classCountMap.size();
	vector<vector<double>> gaussian_mu(nClass);
	vector<vector<double>> gaussian_sigma(nClass);
	for (int n = 0; n < nClass; n++)
		gaussian_mu[n].resize(trainData.cols,0.0);
	for (int n = 0; n < nClass; n++)
		gaussian_sigma[n].resize(trainData.cols,0.0);
	for (int n = 0; n < trainData.rows; n++) {
		int temp_label = (int)(trainLabels.at<float>(n, 0) + 0.5);
		int labelIdx = 0;
		for(map<int,int>::iterator it = classCountMap.begin(); it != classCountMap.end(); ++it) {
			if (it->first==temp_label)
				break;
			else
				labelIdx++;
		}

		for (int m = 0; m < trainData.cols; m++) {
			gaussian_mu[labelIdx][m] += trainData.at<float>(n,m);
		}
	}
	int n = 0;
	for (map<int,int>::iterator it = classCountMap.begin(); it != classCountMap.end(); ++it) {
		for (int m = 0; m < trainData.cols; m++) {
			gaussian_mu[n][m] /= it->second;
		}
		n++;
	}
	for (int n = 0; n < trainData.rows; n++) {
		int temp_label = (int)(trainLabels.at<float>(n, 0) + 0.5);
		int labelIdx = 0;
		for(map<int,int>::iterator it = classCountMap.begin(); it != classCountMap.end(); ++it) {
			if (it->first==temp_label)
				break;
			else
				labelIdx++;
		}
		for (int m = 0; m < trainData.cols; m++) {
			gaussian_sigma[labelIdx][m] += pow(trainData.at<float>(n,m)-gaussian_mu[labelIdx][m],2);
		}
	}
	n = 0;
	for (map<int,int>::iterator it = classCountMap.begin(); it != classCountMap.end(); ++it) {
		for (int m = 0; m < trainData.cols; m++) {
			gaussian_sigma[n][m] /= (it->second);
			gaussian_sigma[n][m] = sqrt(gaussian_sigma[n][m]);
		}
		n++;
	}

	// probability mat
	vector<Mat> probMat(nClass);
	Mat totalProb = Mat::ones(trainData.rows, nClass, CV_64FC1);
	for (int n = 0; n < nClass; n++) {
		Mat prob_mat = Mat::zeros(trainData.size(), CV_64FC1);
		prob_mat.copyTo(probMat[n]);
		for (int m = 0; m < trainData.rows; m++) {
			for (int k = 0; k < trainData.cols; k++) {
				double mu = gaussian_mu[n][k];
				double sigma = gaussian_sigma[n][k];
				double x = trainData.at<float>(m,k);
				if (sigma!=0) {
					probMat[n].at<double>(m,k) = (1/(sigma*sqrt(2*M_PI)))*exp(-0.5*pow((x-mu)/sigma, 2.0));
					probMat[n].at<double>(m,k) /= (1/(sigma*sqrt(2*M_PI)));
				}
				else
					probMat[n].at<double>(m,k) = 1.0;
				//cout<<probMat[n].at<double>(m,k)<<endl;
				totalProb.at<double>(m,n) *= probMat[n].at<double>(m,k);
			}
			//cout<<totalProb.at<double>(m,n)<<endl;
		}
	}

	// generate combinations
	vector<vector<int>> comb_set;
	vector<int> input_num;
	for(map<int,int>::iterator it = classCountMap.begin(); it != classCountMap.end(); ++it) {
		input_num.push_back(it->first);
	}
	for (int n = 1; n < nClass; n++) {
		vector<vector<int>> buff;
		vector<vector<int>> current_set = combnk(buff, input_num, n);
		for (int m = 0; m < current_set.size(); m++)
			comb_set.push_back(current_set[m]);
	}

	// compute error mat
	int comb_num = comb_set.size();
	Mat errMat1 = Mat::zeros(trainData.rows, comb_num, CV_32SC1);
	Mat errMat2 = Mat::zeros(trainData.rows, comb_num, CV_32SC1);
	vector<int> comb_class_num(comb_num, 0);
	for (int k = 0; k < comb_num; k++) {
		int cand_num = comb_set[k].size();
		for (int n = 0; n < cand_num; n++) {
			int tempClassIdx = comb_set[k][n];
			//map<int,int>::iterator it = classCountMap.begin();
			for(map<int,int>::iterator it = classCountMap.begin(); it != classCountMap.end(); ++it) {
				if (it->first==tempClassIdx)
					comb_class_num[k] += it->second;
			}
		}
	}
	for (int n = 0; n < trainData.rows; n++) {
		vector<double> temp_row(totalProb.cols);
		for (int m = 0; m < totalProb.cols; m++)
			temp_row[m] = totalProb.at<double>(n,m);
		int maxIdx = distance(temp_row.begin(), max_element(temp_row.begin(), temp_row.end()));
		map<int,int>::iterator it = classCountMap.begin();
		for (int k = 0; k < maxIdx; k++)
			it++;
		int maxClassIdx = it->first;
		int true_label = (int)(trainLabels.at<float>(n, 0) + 0.5);
		for (int k = 0; k < comb_num; k++) {
			vector<int>::iterator it1, it2;
			it1 = find(comb_set[k].begin(), comb_set[k].end(), maxClassIdx);	//maxIdx
			it2 = find(comb_set[k].begin(), comb_set[k].end(), true_label);
			if ((it1==comb_set[k].end() && it2!=comb_set[k].end()))
				errMat1.at<int>(n,k) = 1;
			if ((it1!=comb_set[k].end() && it2==comb_set[k].end()))
				errMat2.at<int>(n,k) = 1;
		}
	}
	vector<float> totalErr1(comb_num, 0.0);
	vector<float> totalErr2(comb_num, 0.0);
	for (int n = 0; n < comb_num; n++) {
		for (int m = 0; m < trainData.rows; m++) {
			totalErr1[n] += (float)errMat1.at<int>(m,n);
			totalErr2[n] += (float)errMat2.at<int>(m,n);
		}
		totalErr1[n] = totalErr1[n]/comb_class_num[n];
		totalErr2[n] = totalErr2[n]/(trainData.rows-comb_class_num[n]);
	}
	vector<float> totalErr(comb_num);
	for (int n = 0; n < comb_num; n++) {
		totalErr[n] = max(totalErr1[n],totalErr2[n]);
	}
	int bestSepIdx = distance(totalErr.begin(), min_element(totalErr.begin(), totalErr.end()));

	// separate two sets
	Mat posTrainData, posTrainLabels;
	Mat negTrainData, negTrainLabels;
	Mat svmLabels;
	for(size_t i = 0; i < trainData.rows; ++i){
		Mat data = trainData.row(i);
		int label = (int)(trainLabels.at<float>(i, 0) + 0.5);
		vector<int>::iterator it1 = find(comb_set[bestSepIdx].begin(), comb_set[bestSepIdx].end(), label);
		if (it1!=comb_set[bestSepIdx].end()) {
			posTrainData.push_back(data);
			posTrainLabels.push_back((float)label);
			svmLabels.push_back(1.0f);
		}
		else{
			negTrainData.push_back(data);
			negTrainLabels.push_back((float)label);
			svmLabels.push_back(-1.0f);
		}
	}

	// recursion : cluster by EM algorithm
	/*CvEMParams emParams;
	emParams.nclusters = 2;
	emParams.cov_mat_type = CvEM::COV_MAT_DIAGONAL;
	emParams.start_step = CvEM::START_AUTO_STEP;
	emParams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1000, FLT_EPSILON);

	CvEM em;
	Mat clusterLabels;
	em.train(trainData, Mat(), emParams, &clusterLabels);

	// count the data number for each species in both clusters
	map<int, int> posCountMap;
	for(it = classCountMap.begin(); it != classCountMap.end(); ++it)
		posCountMap[it->first] = 0;

	for(size_t i = 0; i < clusterLabels.rows; ++i){
		int label = (int)(trainLabels.at<float>(i, 0) + 0.5);
		int cluster = clusterLabels.at<int>(i, 0);
		if(cluster > 0)
			++posCountMap[label];
	}

	map<int, double> posRatio;
	double maxPosRatio = 0.0;
	int argmaxPosRatio = -1;
	double minPosRatio = 1.0;
	int argminPosRatio = -1;
	for(it = classCountMap.begin(); it != classCountMap.end(); ++it){
		int sp = it->first;
		posRatio[sp] = posCountMap[sp] / (double)(classCountMap[sp]);
		if(posRatio[sp] > maxPosRatio){
			maxPosRatio = posRatio[sp];
			argmaxPosRatio = sp;
		}
		if(posRatio[sp] < minPosRatio){
			minPosRatio = posRatio[sp];
			argminPosRatio = sp;
		}
	}

	// separate data into positive and negative clusters
	Mat posTrainData, posTrainLabels;
	Mat negTrainData, negTrainLabels;
	Mat svmLabels;
	for(size_t i = 0; i < trainData.rows; ++i){
		Mat data = trainData.row(i);
		int label = (int)(trainLabels.at<float>(i, 0) + 0.5);

		if(label != argminPosRatio && posRatio[label] > 0.5 || label == argmaxPosRatio){
			posTrainData.push_back(data);
			posTrainLabels.push_back((float)label);
			svmLabels.push_back(1.0f);
		}
		else{
			negTrainData.push_back(data);
			negTrainLabels.push_back((float)label);
			svmLabels.push_back(-1.0f);
		}
	}*/

	// TODO: change the order of training data
	//
	// In old versions of LIBSVM, the first training instance is always
	// treated as +1 (decision value > 0) regardless of the provided label.
	// As a result, the sign of decision values and predicted labels
	// are reversed if you give -1 data before +1 data.
	// Ref: http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f430

	// create a class hierarchy node and train its SVM
	if(_hierarchy.size() <= id)
		_hierarchy.resize(id + 1, ClassHierarchyNode(-1));
	_hierarchy[id].setID(id);
	_hierarchy[id].trainSVM(trainData, svmLabels);

	clusterClassesRecursive(posTrainData, posTrainLabels, 2*id);

	clusterClassesRecursive(negTrainData, negTrainLabels, 2*id + 1);

}

void ClassHierarchy::loadModel(const string& filename)
{
	FileStorage fs (filename, FileStorage::READ);
	FileNode fn;

	fn = fs["num_classes"];
	fn >> _nClasses;

	// Since the implicit copy constructor of OpenCV class SVM does not
	// function, we use an awkward get-around to read the class	hierarchy
	// nodes by reading the nodes to a temporary vector, keeping track
	// of each node's ID, and re-reading each node to its proper position
	// in the member vector.

	fn = fs["hier_part_classifier"];
	vector<ClassHierarchyNode> tempHierarchy;
	fn >> tempHierarchy;

	int maxID = 0;
	for(size_t i = 0; i < tempHierarchy.size(); ++i){
		int nodeID = tempHierarchy[i].getID();
		if(nodeID > maxID)
			maxID = nodeID;
	}

	_hierarchy.clear();
	_hierarchy.resize(maxID + 1, ClassHierarchyNode());
	FileNodeIterator it = fn.begin();
	for(size_t i = 0; it != fn.end(); ++i){
		int nodeID = tempHierarchy[i].getID();
		it >> _hierarchy[nodeID];
	}

	fs.release();
}

void ClassHierarchy::saveModel(const string& filename)
{
	FileStorage fs (filename, FileStorage::WRITE);

	fs << "num_classes" << _nClasses;

	fs << "hier_part_classifier" << "[";
	for(int i = 0; i < _hierarchy.size(); ++i){
		if(_hierarchy[i].getID() == -1)
			continue;
		fs << _hierarchy[i];
	}
	fs << "]";

	fs.release();
}
