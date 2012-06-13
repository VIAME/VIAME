/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_H
#define VISTK_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_H

/**
 * \file function_process.h
 *
 * \brief Macros for declaring processes which wrap functions.
 */

/**
 * \def CLASS_NAME
 *
 * \brief The class name of a function process given the base name.
 *
 * \param cls The name of the class.
 */
#define CLASS_NAME(cls) function_##cls##_process
/**
 * \def CLASS_DTOR
 *
 * \brief The destructor name of a function process given the base name.
 *
 * \param cls The name of the class.
 */
#define CLASS_DTOR(cls) ~function_##cls##_process

/**
 * \def DECLARE_FUNCTION_PROCESS
 *
 * \brief Declares a class which calls a function when stepped.
 *
 * \param name The base name of the class.
 */
#define DECLARE_FUNCTION_PROCESS(name)               \
class VISTK_NO_EXPORT CLASS_NAME(name)               \
  : public vistk::process                            \
{                                                    \
  public:                                            \
    CLASS_NAME(name)(vistk::config_t const& config); \
    CLASS_DTOR(name)();                              \
  protected:                                         \
    void _configure();                               \
    void _step();                                    \
  private:                                           \
    class priv;                                      \
    boost::scoped_ptr<priv> d;                       \
}

#endif // VISTK_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_H
