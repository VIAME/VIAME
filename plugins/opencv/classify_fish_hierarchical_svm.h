/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Hierarchical SVM Fish Classifier
 *
 * Implementation of a hierarchical fish species classifier using SVMs.
 * Based on original UW Predictor code by Meng-Che Chuang, University of Washington.
 */

#ifndef VIAME_CLASSIFY_FISH_HIERARCHICAL_SVM_H
#define VIAME_CLASSIFY_FISH_HIERARCHICAL_SVM_H

#include "viame_opencv_export.h"

#include <vital/algo/refine_detections.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <sstream>

#if defined(_MSC_VER) && _MSC_VER < 1900
#include <stdarg.h>
#define snprintf c99_snprintf
#define vsnprintf c99_vsnprintf
inline int c99_vsnprintf(char *outBuf, size_t size, const char *format, va_list ap)
{
  int count = -1;
  if (size != 0)
    count = _vsnprintf_s(outBuf, size, _TRUNCATE, format, ap);
  if (count == -1)
    count = _vscprintf(format, ap);
  return count;
}
inline int c99_snprintf(char *outBuf, size_t size, const char *format, ...)
{
  int count;
  va_list ap;
  va_start(ap, format);
  count = c99_vsnprintf(outBuf, size, format, ap);
  va_end(ap);
  return count;
}
#endif

namespace viame {

// Math utility macros and constants
#ifndef M_PI
#define M_PI 3.1415926535897932384626433
#endif

const double PI_ = 3.141592654;
const float body_ratio[5] = {5.5f/6.5f, 4.8f/5.8f, 0.8f, 5.0f/6.0f, 5.2f/6.2f};

// Source image type enumeration
enum sourceImgType { SOURCE_NONE, SOURCE_UNRECTIFIED, SOURCE_RECTIFIED, SOURCE_BOTH };

#ifndef _CAMSRC_
#define _CAMSRC_
enum cameraSource { STEREO_NONE, STEREO_LEFT, STEREO_RIGHT, STEREO_BOTH };
#endif

// ============================================================================
// FGObject - Foreground Object class for tracking detected objects
// ============================================================================
class VIAME_OPENCV_EXPORT FGObject
{
public:
  double area;

  // unrectified (u) and rectified (r) center, points, height, width, and diagonal
  cv::Point2f uCenter, rCenter, umCenter, rmCenter;
  cv::Point2f uVelocity;
  cv::Point2f uPoints[4], rPoints[4], umPoints[4], rmPoints[4];
  float uHeight, rHeight, uWidth, rWidth;
  float uDiagonal, rDiagonal;
  cv::Rect rRect;

  float angle;

  int trackingNum;
  double cumulativeCost;
  int nFrames;
  FGObject* stereoMatch;
  FGObject* prevMatch;
  FGObject* nextMatch;

  cv::Scalar rectColor;

  std::vector<cv::Point> contour;
  bool partialOut;

  cv::Point2f ulMidpoint, urMidpoint;
  cv::Point2f rlMidpoint, rrMidpoint;
  cv::Point2f umlMidpoint, umrMidpoint;

  std::vector<cv::Point3f> triMidpoints;

  cameraSource camSource;

  cv::Mat histogram;

  FGObject();
  ~FGObject() {}

  sourceImgType getFgSourceType() const { return fgSourceType; }

  void setObjectProperties(double a, float ang, std::vector<cv::Point> cont,
                           cv::Point2f pts[], enum sourceImgType imgType);
  void setRect(const cv::Rect& r);
  void setStereoMatch(FGObject* sMatch, cv::Mat mapX, cv::Mat mapY);
  void setPreviousMatch(FGObject* match);
  void setNextMatch(FGObject* match);
  void setStereoObjectProperties(double area, float angle, std::vector<cv::Point> contour,
                                 cv::Point2f unrectPoints[], cv::Point2f rectPoints[],
                                 cameraSource camSource);
  void setFgSourceType(sourceImgType type) { fgSourceType = type; }

  bool isPartialOut(int width, int height);

private:
  enum sourceImgType fgSourceType;
  int ptStart;
};

// ============================================================================
// FGExtraction - Foreground Extraction class
// ============================================================================
class VIAME_OPENCV_EXPORT FGExtraction
{
public:
  FGExtraction();
  FGExtraction(cv::Mat inImage, cv::Mat inBackground);
  ~FGExtraction() {}

  bool detect(cv::Mat inImage);

  bool detectSimple(cv::Mat inImage, cv::Mat inBackground, double theta);

  void findComponents(cv::Mat src);

  void preprocessFg(cv::Mat& fgMask, double seLength, int thresh);

  void morphErosion(cv::Mat& src, int seLength);

  void morphDilation(cv::Mat& src, int seLength);

  void morphOpening(cv::Mat& src, int seLength);

  void morphClosing(cv::Mat& src, int seLength);

  void morphFilling(cv::Mat& src);

  void updateHistogramModel(cv::Mat& hist) { _histModel = hist.clone(); }

  void ratioHistBackprojection(cv::Mat img, cv::Mat& bp);

  void clearState() {
    _inImage.release();
    _background.release();
    _histModel.release();
    _contours.clear();
    _fgObjects.clear();
  }

  cv::Mat getBg() { return _background; }
  cv::Mat getFg() { return _inImage; }
  cv::Mat getHistModel() { return _histModel; }

  int getNumObjects() { return (int)_fgObjects.size(); }
  FGObject& getObject(int objNum) { return _fgObjects[objNum]; }
  std::vector<FGObject>& getObjects() { return _fgObjects; }

  void setBg(cv::Mat& bg) { _background = bg.clone(); }
  void setFg(cv::Mat& fg) { _inImage = fg.clone(); }

protected:
  cv::Mat _inImage;
  cv::Mat _background;
  cv::Mat _histModel;

  int _dilateSize;
  double _minArea;
  double _maxArea;
  double _minAspRatio;
  double _maxAspRatio;

  std::vector<std::vector<cv::Point>> _contours;
  std::vector<FGObject> _fgObjects;

public:
  void setDilateSize(int i) { _dilateSize = i; }
  void setMinArea(double d) { _minArea = d; }
  void setMaxArea(double d) { _maxArea = d; }
  void setMinAspRatio(double d) { _minAspRatio = d; }
  void setMaxAspRatio(double d) { _maxAspRatio = d; }
};

// ============================================================================
// FeatureExtraction - Feature Extraction class for fish classification
// ============================================================================
class VIAME_OPENCV_EXPORT FeatureExtraction
{
public:
  FeatureExtraction();
  FeatureExtraction(cv::Mat& image, cv::Mat& fgMask);
  ~FeatureExtraction() {}

  void setInputImage(cv::Mat& image, cv::Mat& fgMask);

  // Body Part Detection methods
  void runBodyPartDetection(cv::Mat& dst);
  void findEndPts(cv::Mat& fgMask);
  void css(cv::Mat& curvature, cv::Mat& img, cv::Mat& fgMask);

  // Feature Extraction methods
  cv::Mat runFeatureExtraction(cv::Mat& image, cv::Mat& fgMask, bool doOutput = true);
  cv::Mat runBodyPartsFeatureExtraction(cv::Mat& image, cv::Mat& fgMask, bool useTailFD, bool useHeadFD);

  void gradientHistogram(cv::Mat& image, cv::Mat& mask, cv::Mat& features);
  void gradientCurvatureHistogram(cv::Mat& image, cv::Mat& mask, cv::Mat& features);
  void curvatureHistogram(cv::Mat& image, cv::Mat& mask, cv::Mat& features);
  void contourCurvatureHistogram(cv::Mat& fgMask, cv::Mat& features);

  void partitionFg(cv::Mat& fgMask);
  cv::Mat lbp(const cv::Mat& image, cv::Mat& mask);
  cv::Mat hog(cv::Mat& image, cv::Mat& mask, cv::Size cellSize, cv::Size blockSize, cv::Size winStride);
  cv::Mat fdFromContour(const std::vector<cv::Point>& contour, const std::vector<double>& weights);

  cv::Point getTail() { return _ptTail; }
  cv::Point getHead() { return _ptHead; }

protected:
  cv::Mat _image;
  cv::Mat _fgMask;
  cv::Mat _partitionMask;
  std::vector<cv::Point> _contour;

  cv::Point _ptTail;
  cv::Point _ptHead;

  int _lbpNumBlocks;
  int _hogNumBlocks;

public:
  void setLbpNumBlocks(int i) { _lbpNumBlocks = i; }
  void setHogNumBlocks(int i) { _hogNumBlocks = i; }
};

// ============================================================================
// ClassHierarchyNode - Node in the classification hierarchy tree
// ============================================================================
class VIAME_OPENCV_EXPORT ClassHierarchyNode
{
public:
  ClassHierarchyNode() {
    _nodeID = -1;
    _parentID = -1;
    _nodeType = -1;
    _prob = 0.0;
    _condProb = 0.0;
    _leafID = -1;
  }
  ~ClassHierarchyNode() {}

  void setNodeID(int id) { _nodeID = id; }
  void setParentID(int id) { _parentID = id; }
  void setNodeType(int type) { _nodeType = type; }
  void setProb(double prob) { _prob = prob; }
  void setCondProb(double prob) { _condProb = prob; }
  void setLeafID(int id) { _leafID = id; }
  void setChildren(std::vector<int> children) { _children = children; }
  void addChild(int child) { _children.push_back(child); }

  int getNodeID() const { return _nodeID; }
  int getParentID() const { return _parentID; }
  int getNodeType() const { return _nodeType; }
  double getProb() const { return _prob; }
  double getCondProb() const { return _condProb; }
  int getLeafID() const { return _leafID; }
  const std::vector<int>& getChildren() const { return _children; }
  int getNumChildren() const { return (int)_children.size(); }

  cv::Ptr<cv::ml::SVM> getSVM() { return _svm; }
  void setSVM(cv::Ptr<cv::ml::SVM> svm) { _svm = svm; }

private:
  int _nodeID;
  int _parentID;
  int _nodeType;
  double _prob;
  double _condProb;
  int _leafID;
  std::vector<int> _children;
  cv::Ptr<cv::ml::SVM> _svm;
};

// ============================================================================
// ClassHierarchy - Class Hierarchy for hierarchical classification
// ============================================================================
class VIAME_OPENCV_EXPORT ClassHierarchy
{
public:
  ClassHierarchy();
  ~ClassHierarchy();

  void loadModel(const char* filename);
  void saveModel(const char* filename);

  void train(cv::Mat data, cv::Mat labels);
  double predict(cv::Mat features, std::vector<int>& predictions, std::vector<double>& probabilities);

  int getNumNodes() const { return (int)_nodes.size(); }
  int getNumLeaves() const { return _numLeaves; }
  ClassHierarchyNode& getNode(int idx) { return _nodes[idx]; }

  void setNumLeaves(int n) { _numLeaves = n; }

private:
  std::vector<ClassHierarchyNode> _nodes;
  int _numLeaves;

  void initHierarchy();
  void sigmoidFit(cv::Mat labels, cv::Mat decValues, double& A, double& B);
  double sigmoidPredict(double decValue, double A, double B);

  std::vector<double> _sigmoidA;
  std::vector<double> _sigmoidB;
};

// ============================================================================
// FishSpeciesID - Hierarchical Partial Classifier for fish species
// ============================================================================
class VIAME_OPENCV_EXPORT FishSpeciesID
{
public:
  FishSpeciesID();
  ~FishSpeciesID();

  void loadModel(const char* filename);
  void saveModel(const char* filename);

  int outputFeature(cv::Mat img, cv::Mat img2, cv::Mat& feature);

  void train(cv::Mat data, cv::Mat labels);
  bool predict(cv::Mat img, cv::Mat img2, std::vector<int>& predictions,
               std::vector<double>& probabilities, cv::Mat& fgRect);

  int getDimFeat() { return _dimFeat; }

private:
  int extractFeatures(cv::Mat src, cv::Mat src2, cv::Mat& features,
                      bool useTailFD, bool useHeadFD, cv::Mat& fgRect,
                      cv::Point& shift, cv::Mat& rotateR);

  ClassHierarchy _classHierarchy;
  static int _dimFeat;
  int _count;

public:
  // Segmentation parameters
  int _thresh;
  int _seLength;
  double _minArea;
  double _maxArea;
  double _minAspRatio;
  double _maxAspRatio;
  double _histBP_theta;
};

// ============================================================================
// Utility functions
// ============================================================================
std::vector<std::vector<cv::Point>> extractContours(const cv::Mat& img);
cv::RotatedRect orientedBoundingBox(const std::vector<cv::Point>& contour);
cv::Rect outputTargetImage(const FGObject& obj, cv::InputArray src, cv::InputArray fgSrc,
                           cv::OutputArray dst, cv::OutputArray dstFg,
                           cv::Mat& R, cv::Point& shift);
cv::Point computeOrigCoord(cv::Mat R, cv::Point inputPt);

template<class T>
std::string numToStr(const T& num)
{
  std::ostringstream ss;
  ss << num;
  return ss.str();
}

// ============================================================================
// classify_fish_hierarchical_svm - KWIVER algorithm wrapper
// ============================================================================
class VIAME_OPENCV_EXPORT classify_fish_hierarchical_svm
  : public kwiver::vital::algo::refine_detections
{
public:
  PLUGGABLE_IMPL(
    classify_fish_hierarchical_svm,
    "Hierarchical SVM fish species classifier",
    PARAM_DEFAULT( model_file, std::string,
      "Name of hierarchical SVM model file.", "" )
  )

  virtual ~classify_fish_hierarchical_svm() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const
  {
    return true;
  }

  virtual kwiver::vital::detected_object_set_sptr refine(
    kwiver::vital::image_container_sptr image_data,
    kwiver::vital::detected_object_set_sptr input_dets) const;

private:
  mutable FishSpeciesID m_fish_model;
  mutable bool m_model_loaded = false;
};

} // end namespace viame

#endif /* VIAME_CLASSIFY_FISH_HIERARCHICAL_SVM_H */
