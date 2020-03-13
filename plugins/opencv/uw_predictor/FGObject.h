#ifndef _FGOBJECT_H_
#define _FGOBJECT_H_

#include <string>
#include <vector>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

enum sourceImgType {SOURCE_NONE, SOURCE_UNRECTIFIED, SOURCE_RECTIFIED, SOURCE_BOTH};
#ifndef _CAMSRC_
#define _CAMSRC_
enum cameraSource {STEREO_NONE, STEREO_LEFT, STEREO_RIGHT, STEREO_BOTH};
#endif

class FGObject
{
public:

	double			area;

	// unrectified (u) and rectified (r) center, points, height, width, and diaganol
	Point2f			uCenter, rCenter, umCenter, rmCenter;
	Point2f			uVelocity;
	Point2f         uPoints[4], rPoints[4], umPoints[4], rmPoints[4];
	float			uHeight, rHeight, uWidth, rWidth;
	float           uDiagonal, rDiagonal;
	Rect			rRect;
	
	float           angle;
	
	int				trackingNum;
	double			cumulativeCost;
	int				nFrames;
	FGObject*		stereoMatch;
	FGObject*		prevMatch;
	FGObject*		nextMatch;

	Scalar			rectColor;

	vector<Point>	contour;
	bool			partialOut;
	
	/* unrectified left (ul) midpoint
	   unrectified right (ur)
	   rectified left (rl)
	   rectified right (rr) 
	   unrectified mean left (uml)
	   unrectified mean right (umr)    */
	Point2f			ulMidpoint, urMidpoint;
	Point2f         rlMidpoint, rrMidpoint;
	Point2f         umlMidpoint, umrMidpoint;
	
	// triangulated points [left, right]
	vector<Point3f>	triMidpoints;
	
	cameraSource	camSource;

	Mat		histogram;

	FGObject();
	~FGObject() {}

	// getters
	sourceImgType getFgSourceType() const { return fgSourceType; }

	// setters
	void setObjectProperties(double a, float ang, vector<Point> cont, Point2f pts[], enum sourceImgType imgType);
	void setRect(const Rect& r);
	void setStereoMatch(FGObject* sMatch, Mat mapX, Mat mapY);
	void setPreviousMatch(FGObject* match);
	void setNextMatch(FGObject* match);
	void setStereoObjectProperties(double area, float angle, vector<Point> contour, Point2f unrectPoints[],
								   Point2f rectPoints[], cameraSource camSource);
	void setFgSourceType(sourceImgType type) { fgSourceType = type; }

	// check if the object is out of border
	bool isPartialOut(int width, int height);

private:
	enum sourceImgType	fgSourceType;
	int					ptStart;
};

/*******************************************************************************
* Compound Class: StereoObject
*******************************************************************************/
/*class StereoObject
{
public:
	StereoObject(const FGObject& leftObj, const FGObject& rightObj);
	~StereoObject();

	// get functions
	int getTrackingNum() const { return trackingNum; }
	int getFrameNum() const { return frameNum; }
	StereoObject* getPrevMatch() const { return prevMatch; }
	StereoObject* getNextMatch() const { return nextMatch; }
	FGObject* getLeftObject() const { return leftObject; }
	FGObject* getRightObject() const { return rightObject; }

	// set functions
	void setTrackingNum(int n) { trackingNum = n; }
	void setFrameNum(int n) { frameNum = n; }
	void setPreviousMatch(StereoObject* match) { prevMatch = match; }
	void setNextMatch(StereoObject* match) { nextMatch = match; }
	
	void setFGObjectComponents(const FGObject& leftObj, const FGObject& rightObj);


private:
	int				trackingNum;
	int				frameNum;
	StereoObject*	prevMatch;
	StereoObject*	nextMatch;
	
	FGObject*		leftObject;
	FGObject*		rightObject;
};*/

#endif