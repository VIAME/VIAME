//////////////////////////////////////////////////////////////////////////
//
//  classHierarchy.h
//  Date:   Jun/12/2014
//  Author: Meng-Che Chuang, University of Washington
//
//  This file defines the data structures of class tree and tree nodes
//  for the hierarchical partial classification algorithm.
//

#ifndef _CLASS_HIERARCHY_H_
#define _CLASS_HIERARCHY_H_

#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <map>
#include "util.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

//********** forward declaration *************************************************
class ClassHierarchyNode;
class ClassHierarchy;

//********** class classHierarchyNode ********************************************
class ClassHierarchyNode
{
public:
	ClassHierarchyNode();
	ClassHierarchyNode(int id);
	~ClassHierarchyNode();

	void trainSVM(Mat trainData, Mat labels);
	int predictSVM(Mat sample, double& prob);

	int predictSVM2( Mat sample, double& prob_pos, double& prob_neg );

	int getID() const { return _ID; }
	int getPosClass() const { return _posClass; }
	int getNegClass() const { return _negClass; }

	void setID(int id) { _ID = id; }
	void setPosClass(int n) { _posClass = n; }
	void setNegClass(int n) { _negClass = n; }

	void read(const FileNode& fn);
	void write(FileStorage& fs) const;

private:
	double decFuncMargin();
	pair<double, double> fitSigmoid_old(Mat decVals, Mat labels, int posCount, int negCount);
	pair<double, double> fitSigmoid(Mat decVals, Mat labels, int posCount, int negCount);
	double trainPartial(Mat decVals, Mat labels);
	double trainPartialNew(Mat decVals, Mat labels);

	vector<float> fitProb(Mat decVals, Mat labels, int posCount, int negCount);

	// node ID
	int       _ID;

	// positive class
	int       _posClass;

	// negative class
	int       _negClass;

	// SVM margin (distance between support vector and decision hyperplane)
	double    _margin;

	// indecision threshold used by partial classification
	double    _decThresh;

	// sigmoid coefficients for probabilistic output
	double    _sigA;
	double    _sigB;

	double _mu1, _mu2, _sigma1, _sigma2, _w1, _w2;
	// SVM classifier parameters
	SVMParams _svmParams;

	// SVM classifier
	std::shared_ptr< SVM >       _svm;
};

void write(FileStorage& fs, const string& , const ClassHierarchyNode& x);
void read(const FileNode& fn, ClassHierarchyNode& x, const ClassHierarchyNode& default_val = ClassHierarchyNode());

//********** class HierPartialClassifier *****************************************
class ClassHierarchy
{
public:
	ClassHierarchy();
	~ClassHierarchy();

	void loadModel(const string& filename);
	void saveModel(const string& filename);

	void train(Mat trainData, Mat trainLabels);

	bool predict(Mat sample, vector<int>& predictions, vector<double>& probabilities);
	bool predict2(Mat sample, vector<int>& class_label, vector<vector<double>>& probb);

private:
	void clusterClassesRecursive(Mat trainData, Mat trainLabels, int id);

	int _nClasses;
	vector<ClassHierarchyNode> _hierarchy;
	vector<Mat> _speciesData;
	vector<Mat> _speciesLabels;

};

#endif
