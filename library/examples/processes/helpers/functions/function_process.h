// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_H
#define SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_H

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
class SPROKIT_NO_EXPORT CLASS_NAME(name)               \
  : public sprokit::process                            \
{                                                    \
  public:                                            \
    CLASS_NAME(name)(kwiver::vital::config_block_sptr const& config); \
    CLASS_DTOR(name)();                              \
  protected:                                         \
    void _configure();                               \
    void _step();                                    \
  private:                                           \
    class priv;                                      \
    std::unique_ptr<priv> d;                       \
}

#endif // SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_H
