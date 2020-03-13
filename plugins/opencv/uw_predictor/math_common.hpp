/**
 * Copyright 2011 B. Schauerte. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are 
 * met:
 * 
 *    1. Redistributions of source code must retain the above copyright 
 *       notice, this list of conditions and the following disclaimer.
 * 
 *    2. Redistributions in binary form must reproduce the above copyright 
 *       notice, this list of conditions and the following disclaimer in 
 *       the documentation and/or other materials provided with the 
 *       distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY B. SCHAUERTE ''AS IS'' AND ANY EXPRESS OR 
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 * DISCLAIMED. IN NO EVENT SHALL B. SCHAUERTE OR CONTRIBUTORS BE LIABLE 
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *  
 * The views and conclusions contained in the software and documentation
 * are those of the authors and should not be interpreted as representing 
 * official policies, either expressed or implied, of B. Schauerte.
 */

/** math_commmon
 *  Provide basic math functions and constants
 *
 *  \author B. Schauerte
 *  \email  <schauerte@kit.edu>
 *  \date   2008-2011
 */

#pragma once

#include <math.h>
#include <cmath>

#define ID(x)      (x)

#define SQRT(x)    ( sqrt(x) )
#define SQR(x)     ( (x)*(x) )

#define DIAG(a,b)  ( SQRT(SQR(a) + SQR(b)) )
#define PYTH(a,b)  DIAG(a,b)

#ifndef MIN
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a,b) ((a) < (b) ? (b) : (a))
#endif

#define BETWEEN(X,Y,Z) ((((X) >= (Y)) && ((X) <= (Z))) ? true : false)

/** Return the signum */
#define SGN(x) ((x) > 0 ? 1 : ((x) < 0 ? -1 : 0))

#define _INTERVAL_NORMALIZE(__X,__XMIN,__XMAX) ((__X - __XMIN) / (__XMAX - __XMIN))
#define _INTERVAL_DENORMALIZE(__X,__XMIN,__XMAX) ((__X * (__XMAX - __XMIN)) + __XMIN)

/** ld(x)=log2(x) */
template <typename T>
inline
T log2(const T x)
{
  static const T _inv_clog2 = (T)(1.0 / log(2.0));
  return (log(x) * _inv_clog2);
}

//////////////////////////////////////////////////////////////////////////////
// Basic Array-Array Operations
//////////////////////////////////////////////////////////////////////////////

/** Calculate the min/max and argmin/argmax of the array. */
template <typename T, typename S>
inline bool
MinMaxArray(const T* elements, const S numElements, T& min, T& max, S& argMin, S& argMax)
{
  if (elements != 0 && numElements > 0)
  {
    argMin = 0; argMax = 0;
    min = elements[argMin]; max = elements[argMax];

    for (S i = 1; i < numElements; i++)
    {
      if (elements[i] < min)
      {
        min = elements[i];
        argMin = i;
      }
      else if (elements[i] > max)
      {
        max = elements[i];
        argMax = i;
      }
    }

    return true;
  }
  else
    return false;
}

/** Calculate the min/max and argmin/argmax of the array. */
template <typename T, typename S>
inline bool
MinMaxArray(const T* elements, const S numElements, T& min, T& max)
{
  if (elements != 0 && numElements > 0)
  {
    S argMin = 0, argMax = 0;
    min = elements[argMin]; max = elements[argMax];

    for (S i = 1; i < numElements; i++)
    {
      if (elements[i] < min)
      {
        min = elements[i];
        argMin = i;
      }
      else if (elements[i] > max)
      {
        max = elements[i];
        argMax = i;
      }
    }

    return true;
  }
  else
    return false;
}

/** Add, elementwise (dst[x] = dst[x] + src[x]). */
template <typename T, typename S>
void
AddArray(const T* src, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] += src[i];
}

/** Add, elementwise (dst[x] = src1[x] + src2[x]). */
template <typename T, typename S>
void
AddArrays(const T* src1, const T* src2, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] + src2[i];
}

/** Subtract, elementwise (dst[x] = dst[x] - src[x]). */
template <typename T, typename S>
void
SubArray(const T* src, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] -= src[i];
}

/** Subtract, elementwise (dst[x] = src1[x] - src2[x]). */
template <typename T, typename S>
void
SubArrays(const T* src1, const T* src2, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] - src2[i];
}

/** Multiply, elementwise (dst[x] = dst[x] * src[x]). */
template <typename T, typename S>
void
MulArray(const T* src, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] *= src[i];
}

/** Multiply, elementwise (dst[x] = src1[x] * src2[x]). */
template <typename T, typename S>
void
MulArrays(const T* src1, const T* src2, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] * src2[i];
}

/** Divide, elementwise (dst[x] = dst[x] / src[x]). */
template <typename T, typename S>
void
DivArray(const T* src, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] /= src[i];
}

/** Divide, elementwise (dst[x] = src1[x] / src2[x]). */
template <typename T, typename S>
void
DivArrays(const T* src1, const T* src2, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] / src2[i];
}

/** (Bitwise) OR, elementwise (dst[x] = dst[x] | src[x]). */
template <typename T, typename S>
void
OrArray(const T* src, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] |= src[i];
}

/** (Bitwise) OR, elementwise (dst[x] = src1[x] | src2[x]). */
template <typename T, typename S>
void
OrArrays(const T* src1, const T* src2, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] | src2[i];
}

/** (Bitwise) AND, elementwise (dst[x] = dst[x] & src[x]). */
template <typename T, typename S>
void
AndArray(const T* src, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] &= src[i];
}

/** (Bitwise) AND, elementwise (dst[x] = src1[x] & src2[x]). */
template <typename T, typename S>
void
AndArrays(const T* src1, const T* src2, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] & src2[i];
}

/** (Bitwise) XOR, elementwise (dst[x] = dst[x] ^ src[x]). */
template <typename T, typename S>
void
XorArray(const T* src, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] ^= src[i];
}

/** (Bitwise) XOR, elementwise (dst[x] = src1[x] ^ src2[x]). */
template <typename T, typename S>
void
XorArrays(const T* src1, const T* src2, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] ^ src2[i];
}

/** Minimum, elementwise (dst[x] = min(dst[x], src[x])). */
template <typename T, typename S>
void
MinArray(const T* src, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    if (src[i] < dst[i])
      dst[i] = src[i];
}

/** Minimum, elementwise (dst[x] = min(src1[x], src2[x])). */
template <typename T, typename S>
void
MinArrays(const T* src1, const T* src2, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = MIN(src1[i], src2[i]);
}

/** Maximum, elementwise (dst[x] = max(dst[x], src[x])). */
template <typename T, typename S>
void
MaxArray(const T* src, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    if (src[i] > dst[i])
      dst[i] = src[i];
}

/** Maximum, elementwise (dst[x] = max(src1[x], src2[x])). */
template <typename T, typename S>
void
MaxArrays(const T* src1, const T* src2, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = MAX(src1[i], src2[i]);
}

/** Copy elements. Wrapper for memcpy(..); (dst[x] = src[x]). */
template <typename T, typename S>
void
CopyArray(const T* src, T* dst, const S n)
{
  memcpy((void*)dst, (void*)src, sizeof(T) * n);
}

/** Set elements to a constant value (dst[x] = value). */
template <typename T, typename S>
void
SetArray(T* dst, const T value, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = value;
}

/** Set elements to zero  (dst[x] = 0). */
template <typename T, typename S>
void
ZeroArray(T* dst, const S n)
{
  memset((void*)dst, 0, sizeof(T) * n);
    // @note: memset can be unsafe (i.e. produce results !=0) on some machines/systems, but "normally" works
    //        this depends on the used FP-standard (by Compiler and Platform)
}

//////////////////////////////////////////////////////////////////////////////
// Basic Array-Scalar Operations
//////////////////////////////////////////////////////////////////////////////

/** Add, elementwise (dst[x] = src1[x] + value). */
template <typename T, typename S>
void
AddArrayScalar(const T* src1, const T value, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] + value;
}

/** Subtract, elementwise (dst[x] = src1[x] - value). */
template <typename T, typename S>
void
SubArrayScalar(const T* src1, const T value, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] - value;
}

/** Subtract, elementwise (dst[x] = value - src1[x]). */
template <typename T, typename S>
void
SubArrayScalar(const T value, const T* src1, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = value - src1[i];
}

/** Multiply, elementwise (dst[x] = src1[x] * value). */
template <typename T, typename S>
void
MulArrayScalar(const T* src1, const T value, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] * value;
}

/** Divide, elementwise (dst[x] = src1[x] / value). */
template <typename T, typename S>
void
DivArrayScalar(const T* src1, const T value, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] / value;
}

/** Divide, elementwise (dst[x] = value / src1[x]). */
template <typename T, typename S>
void
DivArrayScalar(const T value, const T* src1, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = value / src1[i];
}

/** (Bitwise) OR, elementwise (dst[x] = src1[x] | value). */
template <typename T, typename S>
void
OrArrayScalar(const T* src1, const T value, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] | value;
}

/** (Bitwise) AND, elementwise (dst[x] = src1[x] & value). */
template <typename T, typename S>
void
AndArrayScalar(const T* src1, const T value, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] & value;
}

/** (Bitwise) XOR, elementwise (dst[x] = src1[x] ^ value). */
template <typename T, typename S>
void
XorArrayScalar(const T* src1, const T value, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = src1[i] ^ value;
}

/** Minimum, elementwise (dst[x] = min(src1[x], value)). */
template <typename T, typename S>
void
MinArrayScalar(const T* src1, const T value, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = MIN(src1[i], value);
}

/** Maximum, elementwise (dst[x] = max(src1[x], value)). */
template <typename T, typename S>
void
MaxArrayScalar(const T* src1, const T value, T* dst, const S n)
{
  for (S i = 0; i < n; i++)
    dst[i] = MAX(src1[i], value);
}

//////////////////////////////////////////////////////////////////////////////
// Basic Operations for Angles
//////////////////////////////////////////////////////////////////////////////

//
// Definitions to work with angles
//
//#define PI 3.1415926535897932384626433
#ifndef M_PI
#define M_PI 3.1415926535897932384626433
#endif
#define PI (M_PI)

#define RAD2DEG(rad) (360.0 * (rad) / (2.0 * PI))
#define DEG2RAD(deg) ((2.0 * PI) * (deg) / 360.0)

/** Truncate the angle (in rad) into the interval [-PI;PI] */
// also possible to use a combination of acos/cos, ...
template <typename T>
inline T TruncateAngle(T angle)
{
  // 1. transform into [-2PI;2PI]
  T tmp = angle / ((T)2.0 * (T)PI);
  if (tmp >= (T)0.0)
    tmp = floor(tmp);
  else
    tmp = ceil(tmp);
  angle -= tmp * (T)2.0 * (T)PI;

  // 2. bring into [-PI;PI]
  if (angle > (T)PI)
  {
    angle = angle - (T)2.0 * (T)PI;
  }
  else
  {
    if (angle < (T)(-1.0 * PI))
      angle = (T)2.0 * (T)PI + angle;
  }

  return angle;
}

/** fmod(...) extension with correct negative behaviour (remainder is always positive), e.g. -1.1 mod 3 = 1.9 */
// not designed for negative denominator -> assumes denominator > 0!
inline double
fmodulus(double numerator, double denominator)
{
  double c = fmod(numerator,denominator);
  return (c > 0 ? c : denominator+c);
}

/** fmod(...) extension with correct negative behaviour (remainder is always positive), e.g. -1.1 mod 3 = 1.9 */
// not designed for negative denominator -> assumes denominator > 0!
inline float
fmodulus(float numerator, float denominator)
{
  float c = fmod(numerator,denominator);
  return (c > 0 ? c : denominator+c);
}

/** Truncate the angle (in rad) into the interval [-PI;PI] */
// faster than "TruncateAngle(...)" on SOME systems/compilers - check if this version is better for your system/implementation!
template <typename T>
inline T TruncateAngle2(T angle)
{
  const T d = fmodulus(angle, (T)(2.0 * PI));
  return (d > PI ? d - 2.0*PI : d);
}

/** Truncate the angle (in rad) into the interval [0;2*PI] */
template <typename T>
inline T TruncateAnglePos(T angle)
{
  return fmodulus(angle, (T)(2.0 * PI));
}

//////////////////////////////////////////////////////////////////////////////
// Binomial Coefficient
//////////////////////////////////////////////////////////////////////////////

/** Calculate n choose k. */
template <typename T>
inline T BinomialCoefficient(T n, T k)
{
  if (k > n)
    return 0;

  if (k > n/2)
    k = n-k; // faster

  double accum = 1;
  for (T i = 0; i++ < k;)
    accum *= (double)(n - k + i) / (double)i;

  return (T)(accum + 0.5); // avoid rounding error
}

/** Calculate n choose k. */
template <typename T>
inline T Choose(T n, T k)
{
  return BinomialCoefficient<T>(n,k);
}

//////////////////////////////////////////////////////////////////////////////
// Basic/Primitive Functions
//////////////////////////////////////////////////////////////////////////////

//
// Definitions for basic/common functions
//
#define RAMP(x)                ((x) > 0    ? (x) : 0)
#define HEAVISIDE(x)           ((x) > 0    ? 1   : 0)
#define DELTA_KRONECKER(x,y)   ((x) == (y) ? 1   : 0)
#define KRONECKER_DELTA(x,y)   DELTA_KRONECKER(x,y)

/** The Kronecker-Delta-Function, i.e. ((x) == (y) ? 1 : 0) */
template <typename T>
inline T
KroneckerDelta(T x, T y)
{
  return KRONECKER_DELTA(x,y);
}

/** The Kronecker-Delta-Function, i.e. ((x) == (y) ? 1   : 0) */
template <typename T>
inline T
DeltaKronecker(T x, T y)
{
  return KRONECKER_DELTA(x,y);
}

/** The Ramp-Function, i.e. ((x) > 0 ? (x) : 0) */
template <typename T>
inline T
Ramp(T x)
{
  return RAMP(x);
}

/** The Heaviside-Function, i.e. ((x) > 0 ? 1 : 0) */
template <typename T>
inline T
Heaviside(T x)
{
  return HEAVISIDE(x);
}
