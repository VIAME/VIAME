/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_NUMPY_MEMORY_CHUNK_H
#define VISTK_PYTHON_NUMPY_NUMPY_MEMORY_CHUNK_H

#include "numpy-config.h"

#include <vil/vil_memory_chunk.h>

struct PyArrayObject;

class VISTK_PYTHON_NUMPY_NO_EXPORT numpy_memory_chunk
  : public vil_memory_chunk
{
  public:
    numpy_memory_chunk(PyArrayObject* arr);
    ~numpy_memory_chunk();

    void* data();
    void* const_data() const;

    void set_size(unsigned long n, vil_pixel_format format);
  private:
    void release();

    PyArrayObject* m_arr;
};

#endif // VISTK_PYTHON_NUMPY_NUMPY_MEMORY_CHUNK_H
