#include "FGObject.h"
#include "parameters.h"

//extern Parameters param;

/*

add props for image data and disparity map. these would be rois around the bounding box
block matching should look for source block by looking at the top, middle, and bottom
of the bounding box and taking the block with the greatest sum of pixels. Should BM be
done on a rotated ROI (so the bounding box is oriented with the axes)?

The disparity maps for a stereo pair could be subtracted and the result thresholded
*/


FGObject::FGObject()
{
	// initialize an empty FG object object
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
	uCenter = Point2f(0,0);
	rCenter = Point2f(0,0);
	rmCenter = Point2f(0,0);
	umCenter = Point2f(0,0);
	ulMidpoint = Point2f(0,0);
	urMidpoint = Point2f(0,0);
	rlMidpoint = Point2f(0,0);
	rrMidpoint = Point2f(0,0);
	umlMidpoint = Point2f(0,0);
	umrMidpoint = Point2f(0,0);
	for (int j = 0; j < 4; j++) {
		uPoints[j] = Point(0,0);
		rPoints[j] = Point(0,0);
		rmPoints[j] = Point(0,0);
		umPoints[j] = Point(0,0);
	}

	cumulativeCost = 0;
	nFrames = 1;

	trackingNum = 0;
	rectColor = Scalar(0,0,0);

	partialOut = false;
}


/*******************************************************************************
* Function:      setObjectProperties  
* Description:   set properties for an object in single camera
* Arguments:
	a               -   area
	ang             -   rotation angle
	cont            -   contour
	pts             -   points of the oriented bounding box
	imgType         -   state whether the source image is rectified or not
	                    {SOURCE_NONE, SOURCE_UNRECTIFIED, SOURCE_RECTIFIED, SOURCE_BOTH}
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void FGObject::setObjectProperties(double a, float ang, vector<Point> cont, Point2f pts[], sourceImgType imgType)
{

	float		ulxsum = 0, ulysum = 0, rlxsum = 0, rlysum = 0;
	float		urxsum = 0, urysum = 0, rrxsum = 0, rrysum = 0;

	area = a;
	angle = ang;
	contour = cont;
	
	//  Ensure that the points are ordered clockwise starting in the lower left corner.
	//    It appears that minAreaRect assigns the first point to the point with the greatest y value
	//    (note that the image origin is in the UPPER LEFT CORNER). The points are always returned
	//    in clockwise order so we just need to determine the starting point.
	ptStart = (pts[0].x > pts[2].x) ? 1 : 0;

	if (imgType == SOURCE_UNRECTIFIED) {

		for (int j = 0; j < 4; j++) {
			int k = (ptStart + j > 3) ? 0 : ptStart + j;
			uPoints[j] = pts[k];

			//  sum up the x and y values - sum left and right points separately
			if (j < 2) {
				//  left points
				ulxsum += uPoints[j].x;
				ulysum += uPoints[j].y;
			} 
			else {
				// right points
				urxsum += uPoints[j].x;
				urysum += uPoints[j].y;
			}

		}

		//  calulate width and height, both rectified and unrectified
		uHeight = sqrt(pow(uPoints[0].x - uPoints[1].x,2) + pow(uPoints[0].y - uPoints[1].y,2));
		uWidth = sqrt(pow(uPoints[2].x - uPoints[1].x,2) + pow(uPoints[2].y - uPoints[1].y,2));

		if(uWidth < uHeight){
			float temp = uWidth;
			uWidth = uHeight;
			uHeight = temp;
		}

		//  calculate center based on box corners (not the blob center of mass as returned by findContours)
		uCenter = Point2f(float((urxsum+ulxsum) / 4.), float((urysum+ulysum) / 4.));

		//  calculate bounding box side midpoints
		ulMidpoint = Point2f(float(ulxsum / 2.), float(ulysum / 2.));
		urMidpoint = Point2f(float(urxsum / 2.), float(urysum / 2.));

		//  calculate diagonal
		uDiagonal = sqrt(pow(uPoints[2].x - uPoints[0].x,2) + pow(uPoints[2].y - uPoints[0].y,2));

		//  update the fgSourceType
		if (fgSourceType == SOURCE_RECTIFIED)
			//  we already have the rectified data, so now we have both
			fgSourceType = SOURCE_BOTH;
		else
			//  we don't have any data (or it's the same)
			fgSourceType = imgType;

	}
	else if (imgType == SOURCE_RECTIFIED) {

		//  copy the corner points in order
		for (int j = 0; j < 4; j++) {
			int k = (ptStart + j > 3) ? 0 : ptStart + j;
			rPoints[j] = pts[k];

			//  sum up the x and y values - sum left and right points separately
			if (j < 2) {
				//  left points
				rlxsum += rPoints[j].x;
				rlysum += rPoints[j].y;
			} 
			else {
				// right points
				rrxsum += rPoints[j].x;
				rrysum += rPoints[j].y;
			}
		}

		// calulate width and height, both rectified and unrectified
		rHeight = sqrt(pow(rPoints[0].x - rPoints[1].x,2) + pow(rPoints[0].y - rPoints[1].y,2));
		rWidth = sqrt(pow(rPoints[2].x - rPoints[1].x,2) + pow(rPoints[2].y - rPoints[1].y,2));

		// calculate center based on box corners (not the blob center of mass as returned by findContours)
		rCenter = Point2f(float((rrxsum+rlxsum) / 4.), float((rrysum+rlysum) / 4.));

		//  calculate bounding box side midpoints
		rlMidpoint = Point2f(float(rlxsum / 2.), float(rlysum / 2.));
		rrMidpoint = Point2f(float(rrxsum / 2.), float(rrysum / 2.));

		// calculate diagonal
		rDiagonal = sqrt(pow(rPoints[2].x - rPoints[0].x,2) + pow(rPoints[2].y - rPoints[0].y,2));

		//  update the fgSourceType
		if (fgSourceType == SOURCE_UNRECTIFIED)
			//  we already have the unrectified data, so now we have both
			fgSourceType = SOURCE_BOTH;
		else
			//  we don't have any data (or it's the same)
			fgSourceType = imgType;
	}

}

/*******************************************************************************
* Function:      setRect
* Description:   set bounding rectangle in rectified image
* Arguments:
	r               -   bounding rectangle
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void FGObject::setRect(const Rect& r)
{
	rRect = r;
}

/*******************************************************************************
* Function:      setStereoMatch
* Description:   set stereo match
* Arguments:
	sMatch          -   matched FG object
	mapX            -   rectification map for x coordinate
	mapY			-	rectification map for y coordinate
	
* Returns:       void
* Comments:
* Revision: 
*******************************************************************************/
void FGObject::setStereoMatch(FGObject* sMatch, Mat mapX, Mat mapY)
{

	float			rmlxsum = 0, rmlysum = 0, rmrxsum = 0, rmrysum = 0;
	float			umlxsum = 0, umlysum = 0, umrxsum = 0, umrysum = 0;

	// assign the references of the matching stereo mates to each of the FG objects
	stereoMatch = sMatch;
	stereoMatch->rectColor = rectColor;

	//  convert maps using Mat_<type> for easy direct element access (only copies headers)
	Mat_<float> _mapX = mapX;
	Mat_<float> _mapY = mapY;

	//  calculate the "mean Y" bounding box
	for (int m = 0; m < 4; m++)
	{
		//  calculate the mean Y value for each of the rectified box points and
		//  simply copy the x values
		rmPoints[m].y = (rPoints[m].y + (*stereoMatch).rPoints[m].y) / 2;
		rmPoints[m].x = rPoints[m].x;

		umPoints[m].y = (uPoints[m].y + (*stereoMatch).uPoints[m].y) / 2;
		umPoints[m].x = uPoints[m].x;

		//  sum up the x and y values - sum left and right points separately
		double d01 = norm(uPoints[0] - uPoints[1]);
		double d03 = norm(uPoints[0] - uPoints[3]);

		if (d01 < d03 && m < 2 || d01 >= d03 && m >= 1 && m <= 2) {
			//  left points
			umlxsum += uPoints[m].x;
			umlysum += uPoints[m].y;
			rmlxsum += rPoints[m].x;
			rmlysum += rPoints[m].y;
		} 
		else /*if (d01 < d03 && m >= 2 || d01 >= d03 && (m < 1 || m > 2))*/ {
			// right points
			umrxsum += uPoints[m].x;
			umrysum += uPoints[m].y;
			rmrxsum += rPoints[m].x;
			rmrysum += rPoints[m].y;
		}
	}

	//  calculate the centers for the "mean Y" bounding boxes
	rmCenter = Point2f(float((rmrxsum + rmlxsum) / 4.), float((rmlysum + rmrysum) / 4.));
	umCenter = Point2f(float((umrxsum + umlxsum) / 4.), float((umlysum + umrysum) / 4.));

	//  calculate unrectified matched bounding box side midpoints
	umlMidpoint = Point2f(float(umlxsum / 2.), float(umlysum / 2.));
	umrMidpoint = Point2f(float(umrxsum / 2.), float(umrysum / 2.));

}

void FGObject::setPreviousMatch(FGObject* match)
{
	// assign the references of the matching stereo mates to each of the FG objects
	this->prevMatch = match;
	match->nextMatch = this;

	// assign the color of the bounding box
	rectColor = match->rectColor;
	stereoMatch->rectColor = match->rectColor;
}

void FGObject::setNextMatch(FGObject* match)
{
	// assign the references of the matching stereo mates to each of the FG objects
	this->nextMatch = match;
	match->prevMatch = this;	

	// assign the color of the bounding box
	match->rectColor = rectColor;
	match->stereoMatch->rectColor = rectColor;
}


void FGObject::setStereoObjectProperties(double a, float ang, vector<Point> cont, Point2f unrectPoints[],
										 Point2f rectPoints[], cameraSource cSource)
{

	float		ulxsum = 0, ulysum = 0, rlxsum = 0, rlysum = 0;
	float		urxsum = 0, urysum = 0, rrxsum = 0, rrysum = 0;

	area = a;
	angle = ang;
	contour = cont;
	fgSourceType = SOURCE_BOTH;
	camSource = cSource;

	//  we reset the stereo match pointer since we're assuming that with new parameters we
	//  have a new target, thus a new potential match. This may not always be the case and
	//  this is a caveat one should be aware of.
	stereoMatch = 0;

	//  Ensure that the points are ordered clockwise starting in the lower left corner.
	//    It appears that minAreaRect assigns the first point to the point with the greatest y value
	//    (note that the image origin is in the UPPER LEFT CORNER). The points are always returned
	//    in clockwise order so we just need to determine the starting point.
	ptStart = (rectPoints[0].x > rectPoints[2].x) ? 1 : 0;

	//  copy the corner points in order
	for (int j = 0; j < 4; j++) {
		int k = (ptStart + j) % 4;
		uPoints[j] = unrectPoints[(uPoints ? j : k)];
		rPoints[j] = rectPoints[k];


		//  sum up the x and y values - sum left and right points separately
		if (j < 2) {
			//  left points
			ulxsum += uPoints[j].x;
			ulysum += uPoints[j].y;
			rlxsum += rPoints[j].x;
			rlysum += rPoints[j].y;
		} 
		else {
			// right points
			urxsum += uPoints[j].x;
			urysum += uPoints[j].y;
			rrxsum += rPoints[j].x;
			rrysum += rPoints[j].y;
		}
	}

	// calulate width and height, both rectified and unrectified
	uHeight = sqrt(pow(uPoints[0].x - uPoints[1].x,2) + pow(uPoints[0].y - uPoints[1].y,2));
	rHeight = sqrt(pow(rPoints[0].x - rPoints[1].x,2) + pow(rPoints[0].y - rPoints[1].y,2));
	uWidth = sqrt(pow(uPoints[2].x - uPoints[1].x,2) + pow(uPoints[2].y - uPoints[1].y,2));
	rWidth = sqrt(pow(rPoints[2].x - rPoints[1].x,2) + pow(rPoints[2].y - rPoints[1].y,2));

	// calculate center based on box corners (not the blob center of mass as returned by findContours)
	rCenter = Point2f(float((rrxsum+rlxsum) / 4.), float((rrysum+rlysum) / 4.));
	uCenter = Point2f(float((urxsum+ulxsum) / 4.), float((urysum+ulysum) / 4.));

	//  calculate bounding box side midpoints
	ulMidpoint = Point2f(float(ulxsum / 2.), float(ulysum / 2.));
	urMidpoint = Point2f(float(urxsum / 2.), float(urysum / 2.));
	rlMidpoint = Point2f(float(rlxsum / 2.), float(rlysum / 2.));
	rrMidpoint = Point2f(float(rrxsum / 2.), float(rrysum / 2.));

	// calculate diagonal
	uDiagonal = sqrt(pow(uPoints[2].x - uPoints[0].x,2) + pow(uPoints[2].y - uPoints[0].y,2));
	rDiagonal = sqrt(pow(rPoints[2].x - rPoints[0].x,2) + pow(rPoints[2].y - rPoints[0].y,2));

}

bool
FGObject::isPartialOut(int width, int height)
{
	for(int i = 0; i < 4; ++i){
		if(uPoints[i].x <= -50 || uPoints[i].x >= 2048+50
		  || uPoints[i].y <= -50 || uPoints[i].y >= 2048+50)
			return true;
	}
	return false;
}

////////////////////////////////////////////////////

/*StereoObject::StereoObject(const FGObject& leftObj, const FGObject& rightObj)
{
	setFGObjectComponents(leftObj, rightObj);
}

StereoObject::~StereoObject()
{
	delete leftObject;
	delete rightObject;
}

void StereoObject::setFGObjectComponents(const FGObject& leftObj, const FGObject& rightObj)
{
	leftObject = new FGObject(leftObj);
	rightObject = new FGObject(rightObj);
}
*/