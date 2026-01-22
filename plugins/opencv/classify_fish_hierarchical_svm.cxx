/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Hierarchical SVM Fish Classifier Implementation
 *
 * Based on original UW Predictor code by Meng-Che Chuang, University of Washington.
 */

#include "classify_fish_hierarchical_svm.h"

#include <arrows/ocv/image_container.h>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace viame {

// Initialize static member
int FishSpeciesID::_dimFeat = 131;

// Global RNG for utility functions
static cv::RNG g_rng(54321);

// ============================================================================
// Utility Function Implementations
// ============================================================================

std::vector<std::vector<cv::Point>> extractContours(const cv::Mat& img)
{
  std::vector<std::vector<cv::Point>> contours;
  cv::Mat tempImg = img.clone();
  cv::findContours(tempImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
  tempImg.release();
  return contours;
}

cv::RotatedRect orientedBoundingBox(const std::vector<cv::Point>& contour)
{
  cv::Mat mask8U = cv::Mat::zeros(2048, 2048, CV_8U);
  std::vector<std::vector<cv::Point>> cs;
  cs.push_back(contour);
  cv::drawContours(mask8U, cs, 0, cv::Scalar(255), -1);

  cv::Mat data = cv::Mat::zeros(2, (int)contour.size(), CV_32F);
  for (int j = 0; j < (int)contour.size(); ++j) {
    data.at<float>(0, j) = (float)contour[j].x;
    data.at<float>(1, j) = (float)contour[j].y;
  }

  cv::RotatedRect orientedBox;
  if (contour.size() <= 2) {
    if (contour.size() == 1) {
      orientedBox.center = contour[0];
    } else {
      orientedBox.center.x = 0.5f * (contour[0].x + contour[1].x);
      orientedBox.center.y = 0.5f * (contour[0].y + contour[1].y);
      double dx = contour[1].x - contour[0].x;
      double dy = contour[1].y - contour[0].y;
      orientedBox.size.width = (float)sqrt(dx * dx + dy * dy);
      orientedBox.size.height = 0;
      orientedBox.angle = (float)(atan2(dy, dx) * 180 / CV_PI);
    }
    return orientedBox;
  }

  cv::PCA pcaObj = cv::PCA(data, cv::noArray(), cv::PCA::DATA_AS_COL);

  cv::Mat result;
  pcaObj.project(data, result);

  float maxU = 0, maxV = 0;
  float minU = 0, minV = 0;

  for (int j = 0; j < result.cols; ++j) {
    float u = result.at<float>(0, j);
    float v = result.at<float>(1, j);
    if (u > 0 && u > maxU)
      maxU = u;
    else if (u < 0 && u < minU)
      minU = u;

    if (v > 0 && v > maxV)
      maxV = v;
    else if (v < 0 && v < minV)
      minV = v;
  }

  float cenU = 0.5f * (maxU + minU);
  float cenV = 0.5f * (maxV + minV);

  cv::Mat cenUVMat = (cv::Mat_<float>(2, 1) << cenU, cenV);
  cv::Mat cenXYMat = pcaObj.backProject(cenUVMat);

  cv::Point cen((int)cenXYMat.at<float>(0, 0), (int)cenXYMat.at<float>(1, 0));

  float width = maxU - minU;
  float height = maxV - minV;

  cv::Mat pc = pcaObj.eigenvectors;
  float pcx = pc.at<float>(0, 0);
  float pcy = pc.at<float>(0, 1);
  float theta = atan2(pcy, pcx) * 180.0f / (float)CV_PI;

  orientedBox.center = cen;
  orientedBox.size = cv::Size2f(width, height);
  orientedBox.angle = theta;
  return orientedBox;
}

cv::Rect outputTargetImage(const FGObject& obj, cv::InputArray src, cv::InputArray fgSrc,
                           cv::OutputArray dst, cv::OutputArray dstFg,
                           cv::Mat& R, cv::Point& shift)
{
  if (src.empty() || fgSrc.empty()) return cv::Rect();
  cv::Mat inImg = src.getMat();
  cv::Mat fgImg = fgSrc.getMat();

  float x_min = (float)inImg.cols, y_min = (float)inImg.rows, x_max = 0, y_max = 0;
  for (int k = 0; k < 4; ++k) {
    if (obj.uPoints[k].x < x_min) x_min = obj.uPoints[k].x;
    if (obj.uPoints[k].y < y_min) y_min = obj.uPoints[k].y;
    if (obj.uPoints[k].x > x_max) x_max = obj.uPoints[k].x;
    if (obj.uPoints[k].y > y_max) y_max = obj.uPoints[k].y;
  }
  cv::Point2f tl(x_min, y_min);
  cv::Point2f br(x_max, y_max);

  tl += 0.8f * (tl - obj.uCenter);
  br += 0.8f * (br + obj.uCenter);
  tl.x = tl.x < 0 ? 0 : (tl.x > inImg.cols - 1 ? (float)(inImg.cols - 1) : tl.x);
  tl.y = tl.y < 0 ? 0 : (tl.y > inImg.rows - 1 ? (float)(inImg.rows - 1) : tl.y);
  br.x = br.x < 0 ? 0 : (br.x > inImg.cols - 1 ? (float)(inImg.cols - 1) : br.x);
  br.y = br.x < 0 ? 0 : (br.y > inImg.rows - 1 ? (float)(inImg.rows - 1) : br.y);

  shift.x = (int)tl.x;
  shift.y = (int)tl.y;
  cv::Rect roiRect(tl, br);
  cv::Mat roiImg = inImg(roiRect);
  cv::Mat roiFgImg = fgImg(roiRect);

  // preserve only the blob with largest area
  std::vector<std::vector<cv::Point>> contours = extractContours(roiFgImg);
  double maxA = 0;
  int argMaxA = 0;
  for (int n = 0; n < (int)contours.size(); ++n) {
    double area = cv::contourArea(contours[n]);
    if (area > maxA) {
      maxA = area;
      argMaxA = n;
    }
  }

  cv::Mat tempFg = cv::Mat::zeros(roiFgImg.size(), roiFgImg.type());
  cv::drawContours(tempFg, contours, argMaxA, cv::Scalar(255), -1);
  roiFgImg = tempFg;

  // rotate target by angle of bounding box
  cv::Point2f roiCenter = obj.uCenter - tl;
  double angle = obj.angle;
  if (angle >= 90) {
    angle = obj.angle - 180;
  }

  R = cv::getRotationMatrix2D(roiCenter, angle, 1.0);

  // determine bounding rectangle
  cv::Rect bbox = cv::RotatedRect(roiCenter, fgImg.size(), (float)angle).boundingRect();
  // adjust transformation matrix
  R.at<double>(0, 2) += bbox.width / 2.0 - roiCenter.x;
  R.at<double>(1, 2) += bbox.height / 2.0 - roiCenter.y;

  cv::Mat rotatedRoiImg, rotatedRoiFgImg;
  cv::warpAffine(roiImg, rotatedRoiImg, R, bbox.size());
  cv::warpAffine(roiFgImg, rotatedRoiFgImg, R, bbox.size());

  rotatedRoiImg.copyTo(dst);
  rotatedRoiFgImg.copyTo(dstFg);

  return roiRect;
}

cv::Point computeOrigCoord(cv::Mat R, cv::Point inputPt)
{
  cv::Mat invR;
  cv::invertAffineTransform(R, invR);

  cv::Mat pt = cv::Mat::zeros(3, 1, CV_64FC1);
  pt.at<double>(0, 0) = inputPt.x;
  pt.at<double>(1, 0) = inputPt.y;
  pt.at<double>(2, 0) = 1.0;

  cv::Mat outPt(2, 1, CV_64FC1);
  outPt = invR * pt;
  cv::Point outputPt;
  outputPt.x = (int)outPt.at<double>(0, 0);
  outputPt.y = (int)outPt.at<double>(1, 0);
  return outputPt;
}

// ============================================================================
// FGObject Implementation
// ============================================================================

FGObject::FGObject()
{
  area = 0;
  angle = 0;
  stereoMatch = 0;
  prevMatch = nextMatch = 0;
  fgSourceType = SOURCE_NONE;
  camSource = STEREO_NONE;
  uHeight = 0;
  rHeight = 0;
  uWidth = 0;
  rWidth = 0;
  uCenter = cv::Point2f(0, 0);
  rCenter = cv::Point2f(0, 0);
  rmCenter = cv::Point2f(0, 0);
  umCenter = cv::Point2f(0, 0);
  ulMidpoint = cv::Point2f(0, 0);
  urMidpoint = cv::Point2f(0, 0);
  rlMidpoint = cv::Point2f(0, 0);
  rrMidpoint = cv::Point2f(0, 0);
  umlMidpoint = cv::Point2f(0, 0);
  umrMidpoint = cv::Point2f(0, 0);
  for (int j = 0; j < 4; j++) {
    uPoints[j] = cv::Point2f(0, 0);
    rPoints[j] = cv::Point2f(0, 0);
    rmPoints[j] = cv::Point2f(0, 0);
    umPoints[j] = cv::Point2f(0, 0);
  }

  cumulativeCost = 0;
  nFrames = 1;

  trackingNum = 0;
  rectColor = cv::Scalar(0, 0, 0);

  partialOut = false;
}

void FGObject::setObjectProperties(double a, float ang, std::vector<cv::Point> cont,
                                   cv::Point2f pts[], enum sourceImgType imgType)
{
  float ulxsum = 0, ulysum = 0, rlxsum = 0, rlysum = 0;
  float urxsum = 0, urysum = 0, rrxsum = 0, rrysum = 0;

  area = a;
  angle = ang;
  contour = cont;

  int ptStart = (pts[0].x > pts[2].x) ? 1 : 0;

  if (imgType == SOURCE_UNRECTIFIED) {
    for (int j = 0; j < 4; j++) {
      int k = (ptStart + j > 3) ? 0 : ptStart + j;
      uPoints[j] = pts[k];

      if (j < 2) {
        ulxsum += uPoints[j].x;
        ulysum += uPoints[j].y;
      } else {
        urxsum += uPoints[j].x;
        urysum += uPoints[j].y;
      }
    }

    uHeight = sqrt(pow(uPoints[0].x - uPoints[1].x, 2) + pow(uPoints[0].y - uPoints[1].y, 2));
    uWidth = sqrt(pow(uPoints[2].x - uPoints[1].x, 2) + pow(uPoints[2].y - uPoints[1].y, 2));

    if (uWidth < uHeight) {
      float temp = uWidth;
      uWidth = uHeight;
      uHeight = temp;
    }

    uCenter = cv::Point2f(float((urxsum + ulxsum) / 4.), float((urysum + ulysum) / 4.));
    ulMidpoint = cv::Point2f(float(ulxsum / 2.), float(ulysum / 2.));
    urMidpoint = cv::Point2f(float(urxsum / 2.), float(urysum / 2.));
    uDiagonal = sqrt(pow(uPoints[2].x - uPoints[0].x, 2) + pow(uPoints[2].y - uPoints[0].y, 2));

    if (fgSourceType == SOURCE_RECTIFIED)
      fgSourceType = SOURCE_BOTH;
    else
      fgSourceType = imgType;
  } else if (imgType == SOURCE_RECTIFIED) {
    for (int j = 0; j < 4; j++) {
      int k = (ptStart + j > 3) ? 0 : ptStart + j;
      rPoints[j] = pts[k];

      if (j < 2) {
        rlxsum += rPoints[j].x;
        rlysum += rPoints[j].y;
      } else {
        rrxsum += rPoints[j].x;
        rrysum += rPoints[j].y;
      }
    }

    rHeight = sqrt(pow(rPoints[0].x - rPoints[1].x, 2) + pow(rPoints[0].y - rPoints[1].y, 2));
    rWidth = sqrt(pow(rPoints[2].x - rPoints[1].x, 2) + pow(rPoints[2].y - rPoints[1].y, 2));
    rCenter = cv::Point2f(float((rrxsum + rlxsum) / 4.), float((rrysum + rlysum) / 4.));
    rlMidpoint = cv::Point2f(float(rlxsum / 2.), float(rlysum / 2.));
    rrMidpoint = cv::Point2f(float(rrxsum / 2.), float(rrysum / 2.));
    rDiagonal = sqrt(pow(rPoints[2].x - rPoints[0].x, 2) + pow(rPoints[2].y - rPoints[0].y, 2));

    if (fgSourceType == SOURCE_UNRECTIFIED)
      fgSourceType = SOURCE_BOTH;
    else
      fgSourceType = imgType;
  }
}

void FGObject::setRect(const cv::Rect& r)
{
  rRect = r;
}

void FGObject::setStereoMatch(FGObject* sMatch, cv::Mat mapX, cv::Mat mapY)
{
  float rmlxsum = 0, rmlysum = 0, rmrxsum = 0, rmrysum = 0;
  float umlxsum = 0, umlysum = 0, umrxsum = 0, umrysum = 0;

  stereoMatch = sMatch;
  stereoMatch->rectColor = rectColor;

  cv::Mat_<float> _mapX = mapX;
  cv::Mat_<float> _mapY = mapY;

  for (int m = 0; m < 4; m++) {
    rmPoints[m].y = (rPoints[m].y + (*stereoMatch).rPoints[m].y) / 2;
    rmPoints[m].x = rPoints[m].x;

    umPoints[m].y = (uPoints[m].y + (*stereoMatch).uPoints[m].y) / 2;
    umPoints[m].x = uPoints[m].x;

    double d01 = cv::norm(uPoints[0] - uPoints[1]);
    double d03 = cv::norm(uPoints[0] - uPoints[3]);

    if ((d01 < d03 && m < 2) || (d01 >= d03 && m >= 1 && m <= 2)) {
      umlxsum += uPoints[m].x;
      umlysum += uPoints[m].y;
      rmlxsum += rPoints[m].x;
      rmlysum += rPoints[m].y;
    } else {
      umrxsum += uPoints[m].x;
      umrysum += uPoints[m].y;
      rmrxsum += rPoints[m].x;
      rmrysum += rPoints[m].y;
    }
  }

  rmCenter = cv::Point2f(float((rmrxsum + rmlxsum) / 4.), float((rmlysum + rmrysum) / 4.));
  umCenter = cv::Point2f(float((umrxsum + umlxsum) / 4.), float((umlysum + umrysum) / 4.));
  umlMidpoint = cv::Point2f(float(umlxsum / 2.), float(umlysum / 2.));
  umrMidpoint = cv::Point2f(float(umrxsum / 2.), float(umrysum / 2.));
}

void FGObject::setPreviousMatch(FGObject* match)
{
  this->prevMatch = match;
  match->nextMatch = this;
  rectColor = match->rectColor;
  stereoMatch->rectColor = match->rectColor;
}

void FGObject::setNextMatch(FGObject* match)
{
  this->nextMatch = match;
  match->prevMatch = this;
  match->rectColor = rectColor;
  match->stereoMatch->rectColor = rectColor;
}

void FGObject::setStereoObjectProperties(double a, float ang, std::vector<cv::Point> cont,
                                         cv::Point2f unrectPoints[], cv::Point2f rectPoints[],
                                         cameraSource cSource)
{
  float ulxsum = 0, ulysum = 0, rlxsum = 0, rlysum = 0;
  float urxsum = 0, urysum = 0, rrxsum = 0, rrysum = 0;

  area = a;
  angle = ang;
  contour = cont;
  fgSourceType = SOURCE_BOTH;
  camSource = cSource;
  stereoMatch = 0;

  int ptStart = (rectPoints[0].x > rectPoints[2].x) ? 1 : 0;

  for (int j = 0; j < 4; j++) {
    int k = (ptStart + j) % 4;
    uPoints[j] = unrectPoints[(uPoints ? j : k)];
    rPoints[j] = rectPoints[k];

    if (j < 2) {
      ulxsum += uPoints[j].x;
      ulysum += uPoints[j].y;
      rlxsum += rPoints[j].x;
      rlysum += rPoints[j].y;
    } else {
      urxsum += uPoints[j].x;
      urysum += uPoints[j].y;
      rrxsum += rPoints[j].x;
      rrysum += rPoints[j].y;
    }
  }

  uHeight = sqrt(pow(uPoints[0].x - uPoints[1].x, 2) + pow(uPoints[0].y - uPoints[1].y, 2));
  rHeight = sqrt(pow(rPoints[0].x - rPoints[1].x, 2) + pow(rPoints[0].y - rPoints[1].y, 2));
  uWidth = sqrt(pow(uPoints[2].x - uPoints[1].x, 2) + pow(uPoints[2].y - uPoints[1].y, 2));
  rWidth = sqrt(pow(rPoints[2].x - rPoints[1].x, 2) + pow(rPoints[2].y - rPoints[1].y, 2));

  rCenter = cv::Point2f(float((rrxsum + rlxsum) / 4.), float((rrysum + rlysum) / 4.));
  uCenter = cv::Point2f(float((urxsum + ulxsum) / 4.), float((urysum + ulysum) / 4.));

  ulMidpoint = cv::Point2f(float(ulxsum / 2.), float(ulysum / 2.));
  urMidpoint = cv::Point2f(float(urxsum / 2.), float(urysum / 2.));
  rlMidpoint = cv::Point2f(float(rlxsum / 2.), float(rlysum / 2.));
  rrMidpoint = cv::Point2f(float(rrxsum / 2.), float(rrysum / 2.));

  uDiagonal = sqrt(pow(uPoints[2].x - uPoints[0].x, 2) + pow(uPoints[2].y - uPoints[0].y, 2));
  rDiagonal = sqrt(pow(rPoints[2].x - rPoints[0].x, 2) + pow(rPoints[2].y - rPoints[0].y, 2));
}

bool FGObject::isPartialOut(int width, int height)
{
  for (int i = 0; i < 4; ++i) {
    if (uPoints[i].x <= -50 || uPoints[i].x >= 2048 + 50 ||
        uPoints[i].y <= -50 || uPoints[i].y >= 2048 + 50)
      return true;
  }
  return false;
}

// ============================================================================
// FGExtraction Implementation
// ============================================================================

FGExtraction::FGExtraction()
{
  _dilateSize = 1;
  _minArea = 1000;
  _maxArea = 1e6;
  _minAspRatio = 1.5;
  _maxAspRatio = 30;
}

FGExtraction::FGExtraction(cv::Mat inImage, cv::Mat inBackground)
{
  _inImage = inImage.clone();
  _background = inBackground.clone();
  _dilateSize = 1;
  _minArea = 1000;
  _maxArea = 1e6;
  _minAspRatio = 1.5;
  _maxAspRatio = 30;
}

bool FGExtraction::detect(cv::Mat inImage)
{
  if (_background.empty()) return false;

  _inImage = inImage.clone();

  cv::Mat diffImg;
  cv::absdiff(_inImage, _background, diffImg);

  cv::Mat grayDiff;
  if (diffImg.channels() == 3) {
    cv::cvtColor(diffImg, grayDiff, cv::COLOR_BGR2GRAY);
  } else {
    grayDiff = diffImg;
  }

  cv::Mat binImg;
  cv::threshold(grayDiff, binImg, 25, 255, cv::THRESH_BINARY);

  morphDilation(binImg, _dilateSize);
  morphErosion(binImg, _dilateSize);

  findComponents(binImg);

  return true;
}

bool FGExtraction::detectSimple(cv::Mat inImage, cv::Mat inBackground, double theta)
{
  _inImage = inImage.clone();
  _background = inBackground.clone();

  cv::Mat diffImg;
  cv::absdiff(_inImage, _background, diffImg);

  cv::Mat grayDiff;
  if (diffImg.channels() == 3) {
    cv::cvtColor(diffImg, grayDiff, cv::COLOR_BGR2GRAY);
  } else {
    grayDiff = diffImg;
  }

  cv::Mat binImg;
  cv::threshold(grayDiff, binImg, theta, 255, cv::THRESH_BINARY);

  findComponents(binImg);

  return true;
}

void FGExtraction::findComponents(cv::Mat src)
{
  _contours.clear();
  _fgObjects.clear();

  cv::findContours(src, _contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());

  for (size_t i = 0; i < _contours.size(); ++i) {
    double area = cv::contourArea(_contours[i]);

    if (area < _minArea || area > _maxArea) continue;

    cv::RotatedRect orientedBox = cv::minAreaRect(_contours[i]);

    float aspRatio = orientedBox.size.width / orientedBox.size.height;
    if (aspRatio < 1) aspRatio = 1 / aspRatio;

    if (aspRatio < _minAspRatio || aspRatio > _maxAspRatio) continue;

    cv::Point2f pts[4];
    orientedBox.points(pts);

    FGObject obj;
    obj.setObjectProperties(area, orientedBox.angle, _contours[i], pts, SOURCE_UNRECTIFIED);
    _fgObjects.push_back(obj);
  }
}

void FGExtraction::preprocessFg(cv::Mat& fgMask, double seLength, int thresh)
{
  morphOpening(fgMask, (int)seLength);
  cv::threshold(fgMask, fgMask, thresh, 255, cv::THRESH_BINARY);
}

void FGExtraction::morphErosion(cv::Mat& src, int seLength)
{
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size(2 * seLength + 1, 2 * seLength + 1),
                                               cv::Point(seLength, seLength));
  cv::erode(src, src, element);
}

void FGExtraction::morphDilation(cv::Mat& src, int seLength)
{
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size(2 * seLength + 1, 2 * seLength + 1),
                                               cv::Point(seLength, seLength));
  cv::dilate(src, src, element);
}

void FGExtraction::morphOpening(cv::Mat& src, int seLength)
{
  morphErosion(src, seLength);
  morphDilation(src, seLength);
}

void FGExtraction::morphClosing(cv::Mat& src, int seLength)
{
  morphDilation(src, seLength);
  morphErosion(src, seLength);
}

void FGExtraction::morphFilling(cv::Mat& src)
{
  cv::Mat filled = src.clone();
  cv::floodFill(filled, cv::Point(0, 0), cv::Scalar(255));
  cv::bitwise_not(filled, filled);
  src = src | filled;
}

void FGExtraction::ratioHistBackprojection(cv::Mat img, cv::Mat& bp)
{
  if (_histModel.empty()) {
    bp = cv::Mat::zeros(img.size(), CV_8UC1);
    return;
  }

  cv::Mat hsvImg;
  cv::cvtColor(img, hsvImg, cv::COLOR_BGR2HSV);

  int channels[] = {0, 1};
  int histSize[] = {30, 32};
  float h_ranges[] = {0, 180};
  float s_ranges[] = {0, 256};
  const float* ranges[] = {h_ranges, s_ranges};

  cv::Mat imgHist;
  cv::calcHist(&hsvImg, 1, channels, cv::Mat(), imgHist, 2, histSize, ranges);

  cv::Mat ratioHist = _histModel / (imgHist + 1);

  cv::calcBackProject(&hsvImg, 1, channels, ratioHist, bp, ranges);

  cv::normalize(bp, bp, 0, 255, cv::NORM_MINMAX);
  bp.convertTo(bp, CV_8UC1);
}

// ============================================================================
// FeatureExtraction Implementation
// ============================================================================

FeatureExtraction::FeatureExtraction()
{
  _lbpNumBlocks = 4;
  _hogNumBlocks = 4;
  _ptTail = cv::Point(0, 0);
  _ptHead = cv::Point(0, 0);
}

FeatureExtraction::FeatureExtraction(cv::Mat& image, cv::Mat& fgMask)
{
  _image = image.clone();
  _fgMask = fgMask.clone();
  _lbpNumBlocks = 4;
  _hogNumBlocks = 4;
  _ptTail = cv::Point(0, 0);
  _ptHead = cv::Point(0, 0);
}

void FeatureExtraction::setInputImage(cv::Mat& image, cv::Mat& fgMask)
{
  _image = image.clone();
  _fgMask = fgMask.clone();
}

cv::Mat FeatureExtraction::lbp(const cv::Mat& image, cv::Mat& mask)
{
  cv::Mat lbpImg = cv::Mat::zeros(image.size(), CV_8UC1);

  int neighbors = 8;
  int radius = 1;

  for (int r = radius; r < image.rows - radius; ++r) {
    for (int c = radius; c < image.cols - radius; ++c) {
      if (mask.at<uchar>(r, c) == 0) continue;

      uchar center = image.at<uchar>(r, c);
      uchar code = 0;

      code |= ((image.at<uchar>(r - 1, c - 1) >= center) << 7);
      code |= ((image.at<uchar>(r - 1, c) >= center) << 6);
      code |= ((image.at<uchar>(r - 1, c + 1) >= center) << 5);
      code |= ((image.at<uchar>(r, c + 1) >= center) << 4);
      code |= ((image.at<uchar>(r + 1, c + 1) >= center) << 3);
      code |= ((image.at<uchar>(r + 1, c) >= center) << 2);
      code |= ((image.at<uchar>(r + 1, c - 1) >= center) << 1);
      code |= ((image.at<uchar>(r, c - 1) >= center) << 0);

      lbpImg.at<uchar>(r, c) = code;
    }
  }

  // Compute histogram
  int histSize = 256;
  float range[] = {0, 256};
  const float* histRange = {range};
  cv::Mat hist;
  cv::calcHist(&lbpImg, 1, 0, mask, hist, 1, &histSize, &histRange);

  cv::normalize(hist, hist, 1, 0, cv::NORM_L1);

  return hist.reshape(1, 1);
}

cv::Mat FeatureExtraction::hog(cv::Mat& image, cv::Mat& mask, cv::Size cellSize,
                               cv::Size blockSize, cv::Size winStride)
{
  cv::HOGDescriptor hogDesc;
  hogDesc.winSize = cv::Size((image.cols / cellSize.width) * cellSize.width,
                              (image.rows / cellSize.height) * cellSize.height);
  hogDesc.blockSize = blockSize;
  hogDesc.blockStride = cellSize;
  hogDesc.cellSize = cellSize;
  hogDesc.nbins = 9;

  cv::Mat resizedImg;
  cv::resize(image, resizedImg, hogDesc.winSize);

  std::vector<float> descriptors;
  hogDesc.compute(resizedImg, descriptors);

  cv::Mat hogFeatures(1, (int)descriptors.size(), CV_32F);
  for (size_t i = 0; i < descriptors.size(); ++i) {
    hogFeatures.at<float>(0, (int)i) = descriptors[i];
  }

  return hogFeatures;
}

cv::Mat FeatureExtraction::fdFromContour(const std::vector<cv::Point>& contour,
                                          const std::vector<double>& weights)
{
  if (contour.size() < 4) {
    return cv::Mat::zeros(1, 10, CV_32F);
  }

  // Convert contour to complex representation
  std::vector<cv::Point2f> contourFloat(contour.begin(), contour.end());

  int N = (int)contourFloat.size();

  // Compute centroid
  cv::Point2f centroid(0, 0);
  for (int i = 0; i < N; ++i) {
    centroid += contourFloat[i];
  }
  centroid *= (1.0f / N);

  // Center the contour
  for (int i = 0; i < N; ++i) {
    contourFloat[i] -= centroid;
  }

  // Compute DFT
  cv::Mat complexContour(N, 1, CV_32FC2);
  for (int i = 0; i < N; ++i) {
    complexContour.at<cv::Vec2f>(i, 0) = cv::Vec2f(contourFloat[i].x, contourFloat[i].y);
  }

  cv::Mat dft;
  cv::dft(complexContour, dft);

  // Extract Fourier descriptors (magnitude, normalized)
  int numDescriptors = 10;
  cv::Mat fd(1, numDescriptors, CV_32F);

  float fd1Mag = sqrt(dft.at<cv::Vec2f>(1, 0)[0] * dft.at<cv::Vec2f>(1, 0)[0] +
                      dft.at<cv::Vec2f>(1, 0)[1] * dft.at<cv::Vec2f>(1, 0)[1]);

  if (fd1Mag < 1e-6) fd1Mag = 1e-6f;

  for (int i = 0; i < numDescriptors && i < N / 2; ++i) {
    float mag = sqrt(dft.at<cv::Vec2f>(i + 1, 0)[0] * dft.at<cv::Vec2f>(i + 1, 0)[0] +
                     dft.at<cv::Vec2f>(i + 1, 0)[1] * dft.at<cv::Vec2f>(i + 1, 0)[1]);
    fd.at<float>(0, i) = mag / fd1Mag;
  }

  return fd;
}

void FeatureExtraction::partitionFg(cv::Mat& fgMask)
{
  _partitionMask = fgMask.clone();
}

void FeatureExtraction::gradientHistogram(cv::Mat& image, cv::Mat& mask, cv::Mat& features)
{
  cv::Mat gradX, gradY;
  cv::Sobel(image, gradX, CV_32F, 1, 0, 3);
  cv::Sobel(image, gradY, CV_32F, 0, 1, 3);

  cv::Mat magnitude, angle;
  cv::cartToPolar(gradX, gradY, magnitude, angle, true);

  int histSize = 36;
  float range[] = {0, 360};
  const float* histRange = {range};

  cv::Mat hist;
  cv::calcHist(&angle, 1, 0, mask, hist, 1, &histSize, &histRange);

  cv::normalize(hist, hist, 1, 0, cv::NORM_L1);
  features = hist.reshape(1, 1);
}

void FeatureExtraction::gradientCurvatureHistogram(cv::Mat& image, cv::Mat& mask, cv::Mat& features)
{
  cv::Mat gradX, gradY;
  cv::Sobel(image, gradX, CV_32F, 1, 0, 3);
  cv::Sobel(image, gradY, CV_32F, 0, 1, 3);

  cv::Mat gradXX, gradYY, gradXY;
  cv::Sobel(gradX, gradXX, CV_32F, 1, 0, 3);
  cv::Sobel(gradY, gradYY, CV_32F, 0, 1, 3);
  cv::Sobel(gradX, gradXY, CV_32F, 0, 1, 3);

  cv::Mat curvature = cv::Mat::zeros(image.size(), CV_32F);
  for (int r = 0; r < image.rows; ++r) {
    for (int c = 0; c < image.cols; ++c) {
      float gx = gradX.at<float>(r, c);
      float gy = gradY.at<float>(r, c);
      float gxx = gradXX.at<float>(r, c);
      float gyy = gradYY.at<float>(r, c);
      float gxy = gradXY.at<float>(r, c);

      float denom = pow(gx * gx + gy * gy, 1.5f);
      if (denom > 1e-6) {
        curvature.at<float>(r, c) = (gxx * gy * gy - 2 * gxy * gx * gy + gyy * gx * gx) / denom;
      }
    }
  }

  int histSize = 36;
  float range[] = {-1, 1};
  const float* histRange = {range};

  cv::Mat hist;
  cv::calcHist(&curvature, 1, 0, mask, hist, 1, &histSize, &histRange);

  cv::normalize(hist, hist, 1, 0, cv::NORM_L1);
  features = hist.reshape(1, 1);
}

void FeatureExtraction::curvatureHistogram(cv::Mat& image, cv::Mat& mask, cv::Mat& features)
{
  gradientCurvatureHistogram(image, mask, features);
}

void FeatureExtraction::contourCurvatureHistogram(cv::Mat& fgMask, cv::Mat& features)
{
  std::vector<std::vector<cv::Point>> contours = extractContours(fgMask);
  if (contours.empty()) {
    features = cv::Mat::zeros(1, 36, CV_32F);
    return;
  }

  // Find largest contour
  int maxIdx = 0;
  double maxArea = 0;
  for (size_t i = 0; i < contours.size(); ++i) {
    double area = cv::contourArea(contours[i]);
    if (area > maxArea) {
      maxArea = area;
      maxIdx = (int)i;
    }
  }

  _contour = contours[maxIdx];

  // Compute curvature at each point
  std::vector<float> curvatures;
  int N = (int)_contour.size();
  for (int i = 0; i < N; ++i) {
    cv::Point prev = _contour[(i - 1 + N) % N];
    cv::Point curr = _contour[i];
    cv::Point next = _contour[(i + 1) % N];

    cv::Point2f v1(curr.x - prev.x, curr.y - prev.y);
    cv::Point2f v2(next.x - curr.x, next.y - curr.y);

    float cross = v1.x * v2.y - v1.y * v2.x;
    float dot = v1.x * v2.x + v1.y * v2.y;

    float angle = atan2(cross, dot);
    curvatures.push_back(angle);
  }

  // Build histogram
  int histSize = 36;
  cv::Mat curvMat(curvatures);
  float range[] = {(float)-CV_PI, (float)CV_PI};
  const float* histRange = {range};

  cv::Mat hist;
  cv::calcHist(&curvMat, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

  cv::normalize(hist, hist, 1, 0, cv::NORM_L1);
  features = hist.reshape(1, 1);
}

void FeatureExtraction::findEndPts(cv::Mat& fgMask)
{
  std::vector<std::vector<cv::Point>> contours = extractContours(fgMask);
  if (contours.empty()) return;

  // Find largest contour
  int maxIdx = 0;
  double maxArea = 0;
  for (size_t i = 0; i < contours.size(); ++i) {
    double area = cv::contourArea(contours[i]);
    if (area > maxArea) {
      maxArea = area;
      maxIdx = (int)i;
    }
  }

  _contour = contours[maxIdx];

  if (_contour.size() < 4) return;

  // Find endpoints by looking for max distance from center
  cv::Moments m = cv::moments(fgMask, true);
  cv::Point2f center((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));

  double maxDist1 = 0, maxDist2 = 0;
  cv::Point pt1, pt2;

  for (size_t i = 0; i < _contour.size(); ++i) {
    double dist = cv::norm(cv::Point2f(_contour[i]) - center);
    if (dist > maxDist1) {
      maxDist2 = maxDist1;
      pt2 = pt1;
      maxDist1 = dist;
      pt1 = _contour[i];
    } else if (dist > maxDist2) {
      maxDist2 = dist;
      pt2 = _contour[i];
    }
  }

  // Determine which is head and which is tail based on x position
  if (pt1.x < pt2.x) {
    _ptTail = pt1;
    _ptHead = pt2;
  } else {
    _ptTail = pt2;
    _ptHead = pt1;
  }
}

void FeatureExtraction::css(cv::Mat& curvature, cv::Mat& img, cv::Mat& fgMask)
{
  // Compute CSS (Curvature Scale Space) representation
  std::vector<std::vector<cv::Point>> contours = extractContours(fgMask);
  if (contours.empty()) return;

  int maxIdx = 0;
  double maxArea = 0;
  for (size_t i = 0; i < contours.size(); ++i) {
    double area = cv::contourArea(contours[i]);
    if (area > maxArea) {
      maxArea = area;
      maxIdx = (int)i;
    }
  }

  std::vector<cv::Point>& contour = contours[maxIdx];
  int N = (int)contour.size();

  curvature = cv::Mat::zeros(1, N, CV_32F);

  for (int i = 0; i < N; ++i) {
    cv::Point prev = contour[(i - 1 + N) % N];
    cv::Point curr = contour[i];
    cv::Point next = contour[(i + 1) % N];

    cv::Point2f v1((float)(curr.x - prev.x), (float)(curr.y - prev.y));
    cv::Point2f v2((float)(next.x - curr.x), (float)(next.y - curr.y));

    float cross = v1.x * v2.y - v1.y * v2.x;
    float dot = v1.x * v2.x + v1.y * v2.y;

    curvature.at<float>(0, i) = atan2(cross, dot);
  }
}

void FeatureExtraction::runBodyPartDetection(cv::Mat& dst)
{
  findEndPts(_fgMask);

  dst = _image.clone();
  if (dst.channels() == 1) {
    cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
  }

  cv::circle(dst, _ptHead, 5, cv::Scalar(0, 255, 0), -1);
  cv::circle(dst, _ptTail, 5, cv::Scalar(0, 0, 255), -1);
}

cv::Mat FeatureExtraction::runFeatureExtraction(cv::Mat& image, cv::Mat& fgMask, bool doOutput)
{
  setInputImage(image, fgMask);

  // Extract various features
  cv::Mat grayImg;
  if (image.channels() == 3) {
    cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);
  } else {
    grayImg = image.clone();
  }

  // LBP features
  cv::Mat lbpFeatures = lbp(grayImg, fgMask);

  // Gradient histogram
  cv::Mat gradHist;
  gradientHistogram(grayImg, fgMask, gradHist);

  // Contour curvature histogram
  cv::Mat curvHist;
  contourCurvatureHistogram(fgMask, curvHist);

  // Fourier descriptors
  std::vector<double> weights(10, 1.0);
  findEndPts(fgMask);
  cv::Mat fd = fdFromContour(_contour, weights);

  // Concatenate all features
  cv::Mat features;
  cv::hconcat(lbpFeatures, gradHist, features);
  cv::hconcat(features, curvHist, features);
  cv::hconcat(features, fd, features);

  return features;
}

cv::Mat FeatureExtraction::runBodyPartsFeatureExtraction(cv::Mat& image, cv::Mat& fgMask,
                                                          bool useTailFD, bool useHeadFD)
{
  return runFeatureExtraction(image, fgMask, false);
}

// ============================================================================
// ClassHierarchy Implementation
// ============================================================================

ClassHierarchy::ClassHierarchy()
{
  _numLeaves = 0;
}

ClassHierarchy::~ClassHierarchy()
{
}

void ClassHierarchy::initHierarchy()
{
  // Initialize a default hierarchy (can be customized via loadModel)
  _nodes.clear();
  _sigmoidA.clear();
  _sigmoidB.clear();
}

void ClassHierarchy::loadModel(const char* filename)
{
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "Error: Cannot open model file: " << filename << std::endl;
    return;
  }

  int numNodes = (int)fs["numNodes"];
  _numLeaves = (int)fs["numLeaves"];

  _nodes.resize(numNodes);
  _sigmoidA.resize(numNodes);
  _sigmoidB.resize(numNodes);

  cv::FileNode nodesNode = fs["nodes"];
  int idx = 0;
  for (cv::FileNodeIterator it = nodesNode.begin(); it != nodesNode.end(); ++it, ++idx) {
    cv::FileNode node = *it;

    _nodes[idx].setNodeID((int)node["nodeID"]);
    _nodes[idx].setParentID((int)node["parentID"]);
    _nodes[idx].setNodeType((int)node["nodeType"]);
    _nodes[idx].setLeafID((int)node["leafID"]);

    std::vector<int> children;
    node["children"] >> children;
    _nodes[idx].setChildren(children);

    _sigmoidA[idx] = (double)node["sigmoidA"];
    _sigmoidB[idx] = (double)node["sigmoidB"];

    std::string svmFile = (std::string)node["svmFile"];
    if (!svmFile.empty()) {
      cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svmFile);
      _nodes[idx].setSVM(svm);
    }
  }

  fs.release();
}

void ClassHierarchy::saveModel(const char* filename)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
    return;
  }

  fs << "numNodes" << (int)_nodes.size();
  fs << "numLeaves" << _numLeaves;

  fs << "nodes" << "[";
  for (size_t i = 0; i < _nodes.size(); ++i) {
    fs << "{";
    fs << "nodeID" << _nodes[i].getNodeID();
    fs << "parentID" << _nodes[i].getParentID();
    fs << "nodeType" << _nodes[i].getNodeType();
    fs << "leafID" << _nodes[i].getLeafID();
    fs << "children" << _nodes[i].getChildren();
    fs << "sigmoidA" << _sigmoidA[i];
    fs << "sigmoidB" << _sigmoidB[i];
    fs << "}";
  }
  fs << "]";

  fs.release();
}

void ClassHierarchy::sigmoidFit(cv::Mat labels, cv::Mat decValues, double& A, double& B)
{
  // Platt's sigmoid fitting for probabilistic SVM outputs
  int N = labels.rows;
  double prior1 = 0, prior0 = 0;

  for (int i = 0; i < N; ++i) {
    if (labels.at<int>(i, 0) > 0)
      prior1++;
    else
      prior0++;
  }

  double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
  double loTarget = 1.0 / (prior0 + 2.0);

  std::vector<double> t(N);
  for (int i = 0; i < N; ++i) {
    if (labels.at<int>(i, 0) > 0)
      t[i] = hiTarget;
    else
      t[i] = loTarget;
  }

  A = 0;
  B = log((prior0 + 1.0) / (prior1 + 1.0));

  double fApB, p, q, h11, h22, h21, d1, d2, det, dA, dB, gd, stepsize;
  double fval = 0;

  int maxIter = 100;
  double minStep = 1e-10;
  double sigma = 1e-12;

  for (int iter = 0; iter < maxIter; ++iter) {
    h11 = sigma;
    h22 = sigma;
    h21 = 0;
    d1 = 0;
    d2 = 0;

    for (int i = 0; i < N; ++i) {
      fApB = decValues.at<float>(i, 0) * A + B;
      if (fApB >= 0) {
        p = exp(-fApB) / (1.0 + exp(-fApB));
        q = 1.0 / (1.0 + exp(-fApB));
      } else {
        p = 1.0 / (1.0 + exp(fApB));
        q = exp(fApB) / (1.0 + exp(fApB));
      }
      d2 += (t[i] - p);
      d1 += decValues.at<float>(i, 0) * (t[i] - p);
      h11 += decValues.at<float>(i, 0) * decValues.at<float>(i, 0) * p * q;
      h22 += p * q;
      h21 += decValues.at<float>(i, 0) * p * q;
    }

    det = h11 * h22 - h21 * h21;
    dA = -(h22 * d1 - h21 * d2) / det;
    dB = -(-h21 * d1 + h11 * d2) / det;
    gd = d1 * dA + d2 * dB;

    stepsize = 1;
    while (stepsize >= minStep) {
      double newA = A + stepsize * dA;
      double newB = B + stepsize * dB;
      double newf = 0;

      for (int i = 0; i < N; ++i) {
        fApB = decValues.at<float>(i, 0) * newA + newB;
        if (fApB >= 0)
          newf += t[i] * fApB + log(1 + exp(-fApB));
        else
          newf += (t[i] - 1) * fApB + log(1 + exp(fApB));
      }

      if (newf < fval + 0.0001 * stepsize * gd) {
        A = newA;
        B = newB;
        fval = newf;
        break;
      } else {
        stepsize /= 2.0;
      }
    }

    if (stepsize < minStep) break;
  }
}

double ClassHierarchy::sigmoidPredict(double decValue, double A, double B)
{
  double fApB = decValue * A + B;
  if (fApB >= 0)
    return exp(-fApB) / (1.0 + exp(-fApB));
  else
    return 1.0 / (1.0 + exp(fApB));
}

void ClassHierarchy::train(cv::Mat data, cv::Mat labels)
{
  // Training implementation for hierarchical SVM
  // This is a simplified version; the full implementation would build the hierarchy
  initHierarchy();

  // Create root node with SVM
  ClassHierarchyNode root;
  root.setNodeID(0);
  root.setParentID(-1);
  root.setNodeType(0);

  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);
  svm->setKernel(cv::ml::SVM::RBF);
  svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

  svm->train(data, cv::ml::ROW_SAMPLE, labels);

  root.setSVM(svm);
  _nodes.push_back(root);

  // Compute sigmoid parameters
  cv::Mat decValues;
  svm->predict(data, decValues, cv::ml::StatModel::RAW_OUTPUT);

  double A, B;
  sigmoidFit(labels, decValues, A, B);
  _sigmoidA.push_back(A);
  _sigmoidB.push_back(B);
}

double ClassHierarchy::predict(cv::Mat features, std::vector<int>& predictions,
                               std::vector<double>& probabilities)
{
  predictions.clear();
  probabilities.clear();

  if (_nodes.empty()) return 0.0;

  // Traverse the hierarchy from root
  int currentNode = 0;
  double totalProb = 1.0;

  while (currentNode >= 0 && currentNode < (int)_nodes.size()) {
    ClassHierarchyNode& node = _nodes[currentNode];

    if (node.getNumChildren() == 0) {
      // Leaf node
      predictions.push_back(node.getLeafID());
      probabilities.push_back(totalProb);
      break;
    }

    cv::Ptr<cv::ml::SVM> svm = node.getSVM();
    if (svm.empty()) break;

    float decValue = 0;
    cv::Mat result;
    svm->predict(features, result, cv::ml::StatModel::RAW_OUTPUT);
    decValue = result.at<float>(0, 0);

    double prob = sigmoidPredict(decValue, _sigmoidA[currentNode], _sigmoidB[currentNode]);
    totalProb *= prob;

    // Choose child based on SVM decision
    const std::vector<int>& children = node.getChildren();
    if (children.size() >= 2) {
      currentNode = (decValue > 0) ? children[0] : children[1];
    } else if (children.size() == 1) {
      currentNode = children[0];
    } else {
      break;
    }
  }

  return totalProb;
}

// ============================================================================
// FishSpeciesID Implementation
// ============================================================================

FishSpeciesID::FishSpeciesID()
{
  _count = 0;
  _thresh = 127;
  _seLength = 3;
  _minArea = 1000;
  _maxArea = 1e6;
  _minAspRatio = 1.5;
  _maxAspRatio = 30;
  _histBP_theta = 0.5;
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

int FishSpeciesID::extractFeatures(cv::Mat src, cv::Mat src2, cv::Mat& features,
                                   bool useTailFD, bool useHeadFD, cv::Mat& fgRect,
                                   cv::Point& shift, cv::Mat& rotateR)
{
  FeatureExtraction fe;
  features = fe.runBodyPartsFeatureExtraction(src, src2, useTailFD, useHeadFD);
  return features.cols;
}

int FishSpeciesID::outputFeature(cv::Mat img, cv::Mat img2, cv::Mat& feature)
{
  cv::Point shift;
  cv::Mat rotateR, fgRect;
  return extractFeatures(img, img2, feature, true, true, fgRect, shift, rotateR);
}

void FishSpeciesID::train(cv::Mat data, cv::Mat labels)
{
  _classHierarchy.train(data, labels);
}

bool FishSpeciesID::predict(cv::Mat img, cv::Mat img2, std::vector<int>& predictions,
                            std::vector<double>& probabilities, cv::Mat& fgRect)
{
  cv::Mat features;
  cv::Point shift;
  cv::Mat rotateR;

  int nFeatures = extractFeatures(img, img2, features, true, true, fgRect, shift, rotateR);

  if (nFeatures == 0 || features.empty()) {
    return false;
  }

  _classHierarchy.predict(features, predictions, probabilities);

  return !predictions.empty();
}

// ============================================================================
// classify_fish_hierarchical_svm KWIVER Algorithm Wrapper
// ============================================================================

class classify_fish_hierarchical_svm::priv
{
public:
  priv() {}
  ~priv() {}

  std::string m_model_file;
  FishSpeciesID m_fish_model;
};

classify_fish_hierarchical_svm::classify_fish_hierarchical_svm()
  : d(new priv)
{
}

classify_fish_hierarchical_svm::~classify_fish_hierarchical_svm()
{
}

kwiver::vital::config_block_sptr
classify_fish_hierarchical_svm::get_configuration() const
{
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  config->set_value("model_file", d->m_model_file,
                    "Name of hierarchical SVM model file.");

  return config;
}

void
classify_fish_hierarchical_svm::set_configuration(kwiver::vital::config_block_sptr config)
{
  d->m_model_file = config->get_value<std::string>("model_file");
  d->m_fish_model.loadModel(d->m_model_file.c_str());
}

bool
classify_fish_hierarchical_svm::check_configuration(kwiver::vital::config_block_sptr config) const
{
  return true;
}

kwiver::vital::detected_object_set_sptr
classify_fish_hierarchical_svm::refine(
  kwiver::vital::image_container_sptr image_data,
  kwiver::vital::detected_object_set_sptr input_dets) const
{
  using ocv_container = kwiver::arrows::ocv::image_container;

  auto output_detections = std::make_shared<kwiver::vital::detected_object_set>();

  cv::Mat src = ocv_container::vital_to_ocv( image_data->get_image(),
                                             ocv_container::BGR_COLOR );

  if (src.channels() == 3) {
    cv::cvtColor(src, src, cv::COLOR_RGB2GRAY);
  }

  for (auto det : *input_dets) {
    auto bbox = det->bounding_box();

    cv::Rect roi((int)bbox.min_x(), (int)bbox.min_y(),
                 (int)bbox.width(), (int)bbox.height());
    cv::Mat roi_crop = src(roi);

    std::vector<int> predictions;
    std::vector<double> probabilities;

    cv::Mat segment_chip = ocv_container::vital_to_ocv(
      det->mask()->get_image(), ocv_container::OTHER_COLOR );

    if (segment_chip.channels() == 3) {
      cv::cvtColor(segment_chip, segment_chip, cv::COLOR_RGB2GRAY);
    }

    cv::Mat fg_rect;
    bool success = d->m_fish_model.predict(roi_crop, segment_chip, predictions,
                                           probabilities, fg_rect);

    if (!success) {
      output_detections->add(det);
      continue;
    }

    std::vector<std::string> names;
    for (int i : predictions) {
      names.push_back(std::to_string(i));
    }

    auto dot = std::make_shared<kwiver::vital::detected_object_type>(names, probabilities);

    output_detections->add(
      std::make_shared<kwiver::vital::detected_object>(bbox, 1.0, dot));
  }

  return output_detections;
}

} // end namespace viame
