# Required SVM external dependency

option( KWIVER_ENABLE_SVM
        "Enable SVM arrow for Kwiver"
        OFF
        )

if(KWIVER_ENABLE_SVM)
    find_library(LIBSVM svm)

    if(NOT LIBSVM)
        # Path to svm library
        set(CUSTOM_LIBSVM_PATH )
        if(NOT CUSTOM_LIBSVM_PATH)
            MESSAGE(FATAL_ERROR "SVM LIBRARY NOT FOUND")
        endif()
        include_directories(${CUSTOM_LIBSVM_PATH}/include)
        link_directories(${CUSTOM_LIBSVM_PATH}/lib)
    endif()
endif()
