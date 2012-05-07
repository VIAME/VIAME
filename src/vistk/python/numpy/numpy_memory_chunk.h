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

/**
 * \class numpy_memory_chunk numpy_memory_chunk.h <vistk/python/numpy/numpy_memory_chunk.h>
 *
 * \brief A wrapper for vil's memory_chuck to point to NumPy-managed memory.
 */
class VISTK_PYTHON_NUMPY_NO_EXPORT numpy_memory_chunk
  : public vil_memory_chunk
{
  public:
    /**
     * \brief Constructor.
     *
     * \param arr The NumPy object holding the memory.
     */
    numpy_memory_chunk(PyArrayObject* arr);
    /**
     * \brief Destructor.
     */
    ~numpy_memory_chunk();

    /**
     * \brief The data pointed to by the chunk.
     *
     * \returns The internal data.
     */
    void* data();
    /**
     * \brief The data pointed to by the chunk.
     *
     * \returns The internal data.
     */
    void* const_data() const;

    /**
     * \brief Sets the size of the data pointed to by the chunk.
     *
     * \note After this is called, the NumPy memory is dropped and vil manages
     * the memory instead.
     */
    void set_size(unsigned long n, vil_pixel_format format);
  private:
    void release();

    PyArrayObject* m_arr;
};

#endif // VISTK_PYTHON_NUMPY_NUMPY_MEMORY_CHUNK_H
