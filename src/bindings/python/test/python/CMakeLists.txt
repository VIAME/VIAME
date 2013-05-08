project(sprokit_python_test_python)

set(no_install TRUE)

add_subdirectory(modules)

sprokit_create_python_init(sprokit/test/python)
