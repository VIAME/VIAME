# makefile for linux build 
PROJ_DIR=$(realpath ./)
INSTALL_DIR=${PROJ_DIR}/local
DEP_DIR=${PROJ_DIR}/dependencies
VIAME_DIR=${PROJ_DIR}/../VIAME

BOOST_ROOT=${INSTALL_DIR}
OpenCV_DIR=/usr/local/share/OpenCV

# To use a name different from the default $(USER) to access git-open.sarnoff.com, use
#                 make GIT_USER=name -f linux.mk
GIT_USER = $(USER)

##################################################################
all:	build_dir build_dependencies build_fishtrack build_PythonModule

cleanall:	clear_dependencies clear_fishtrack
	rm -rf ${INSTALL_DIR}

pullall:
	@for mod in db feature_association FeatureDescriptorLib rknave vp vtcmake vtlapack ; do\
		echo pull $$mod...;\
		cd ${DEP_DIR}/$$mod && git pull;\
	done
	echo pull fishtrack...
	cd ${PROJ_DIR} && git pull

##################################################################
build_dir:	${DEP_DIR} ${INSTALL_DIR}

##################################################################
build_dependencies:	build_vtcmake build_vtlapack build_vp build_rknave build_boost build_opencv build_FeatureDescriptorLib build_db build_feature_association build_boost_numpy build_selective_search build_openblas build_caffe

clear_dependencies: clear_vtlapack clear_vp clear_rknave clear_FeatureDescriptorLib clear_db clear_feature_association

##################################################################
build_vtcmake:	${INSTALL_DIR}/vtcmake

${INSTALL_DIR}/vtcmake:	${DEP_DIR}/vtcmake 
	cd ${DEP_DIR}/vtcmake; cmake -DINSTALL_PATH=${INSTALL_DIR} -P install.cmake

${DEP_DIR}/vtcmake:
	echo GIT_USER=$(GIT_USER)
	cd ${DEP_DIR}; git clone $(GIT_USER)@git-open.sarnoff.com:/scm/vision/vtcmake

##################################################################
build_vtlapack:	build_vtcmake ${INSTALL_DIR}/lib/libvtlapack.a

clear_vtlapack:
	-rm -f ${INSTALL_DIR}/lib/libvtlapack.a
	-rm -r ${DEP_DIR}/vtlapack/build

${INSTALL_DIR}/lib/libvtlapack.a:	${DEP_DIR}/vtlapack
	mkdir -p ${DEP_DIR}/vtlapack/build
	cd ${DEP_DIR}/vtlapack/build; cmake -Dvtcmake_DIR=${INSTALL_DIR}/vtcmake -DUSE_MKL=no  -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} ..
	cd ${DEP_DIR}/vtlapack/build; make install

${DEP_DIR}/vtlapack:
	echo GIT_USER=$(GIT_USER)
	cd ${DEP_DIR};git clone $(GIT_USER)@git-open.sarnoff.com:/scm/vision/vtlapack
	cd ${DEP_DIR}/vtlapack; git checkout B_ZYNQ

##################################################################
build_vp:	build_vtcmake build_vtlapack ${INSTALL_DIR}/lib/libvpfilter.a

clear_vp:
	-rm -f ${INSTALL_DIR}/lib/libvpfilter.a
	-rm -rf ${DEP_DIR}/vp/build

${INSTALL_DIR}/lib/libvpfilter.a:	${DEP_DIR}/vp
	mkdir -p ${DEP_DIR}/vp/build
	cd ${DEP_DIR}/vp/build; cmake -Dvtcmake_DIR=${INSTALL_DIR}/vtcmake -DUSE_MKL=no -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} -DVP_STATIC=true -DCMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu ..
	cd ${DEP_DIR}/vp/build; make install

${DEP_DIR}/vp:
	echo GIT_USER=$(GIT_USER)
	cd ${DEP_DIR};git clone $(GIT_USER)@git-open.sarnoff.com:/scm/vision/vp

##################################################################
build_rknave:	build_vtcmake build_vtlapack ${INSTALL_DIR}/lib/librknave8s.a

clear_rknave:
	-rm -f ${INSTALL_DIR}/lib/librknave8s.a
	-rm -r ${DEP_DIR}/rknave/build

${INSTALL_DIR}/lib/librknave8s.a:	${DEP_DIR}/rknave
	mkdir -p ${DEP_DIR}/rknave/build
	cd ${DEP_DIR}/rknave/build; cmake -Dvtcmake_DIR=${INSTALL_DIR}/vtcmake -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} ..
	cd ${DEP_DIR}/rknave/build; make install

${DEP_DIR}/rknave:
	echo GIT_USER=$(GIT_USER)
	cd ${DEP_DIR};git clone $(GIT_USER)@git-open.sarnoff.com:/scm/vision/rknave

##################################################################
BOOST_VER=58
build_boost: ${INSTALL_DIR}/lib/libboost_system.a

clear_boost:
	-rm -f ${INSTALL_DIR}/lib/libboost_*
	-rm -r ${DEP_DIR}/boost_1_${BOOST_VER}_0

${INSTALL_DIR}/lib/libboost_system.a:	${DEP_DIR}/boost_1_${BOOST_VER}_0
	cd ${DEP_DIR}/boost_1_${BOOST_VER}_0; ./bootstrap.sh --prefix=${INSTALL_DIR} --with-python=python2.7
	cd ${DEP_DIR}/boost_1_${BOOST_VER}_0; ./b2 -s NO_BZIP2=1 cxxflags=-fPIC install
	#cd ${INSTALL_DIR}/lib; cp libboost_python.a libboost_python-py27.a; cp libboost_python.so libboost_python-py27.so; cp libboost_python.so.1.55.0 libboost_python-py27.so.1.55.0

${DEP_DIR}/boost_1_${BOOST_VER}_0:
	cd ${DEP_DIR};wget -c 'http://sourceforge.net/projects/boost/files/boost/1.${BOOST_VER}.0/boost_1_${BOOST_VER}_0.zip'
	cd ${DEP_DIR}; unzip boost_1_${BOOST_VER}_0.zip
#	rcp $(GIT_USER)@git-open.sarnoff.com:/scm/vision/evsdk/opensw/boost_1_60_0.zip ${DEP_DIR}/

##################################################################
build_FeatureDescriptorLib:	${INSTALL_DIR}/lib/libFeatureDescriptor.a

clear_FeatureDescriptorLib:
	-rm -f ${INSTALL_DIR}/lib/libFeatureDescriptor.a
	-rm -r ${DEP_DIR}/FeatureDescriptorLib/build

${INSTALL_DIR}/lib/libFeatureDescriptor.a:	${DEP_DIR}/FeatureDescriptorLib
	mkdir -p ${DEP_DIR}/FeatureDescriptorLib/build
	cd ${DEP_DIR}/FeatureDescriptorLib/build;cmake -DQT_EXCLUDE=true -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}  -DFEATURE_DESCRIPTOR_SHARED_LIB=OFF -DBOOST_ROOT=${BOOST_ROOT} -DOpenCV_DIR=${OpenCV_DIR} ..
	cd ${DEP_DIR}/FeatureDescriptorLib/build; make install

${DEP_DIR}/FeatureDescriptorLib:
	echo GIT_USER=$(GIT_USER)
	cd ${DEP_DIR};git clone $(GIT_USER)@git-open.sarnoff.com:/scm/vision/FeatureDescriptorLib

##################################################################
build_db:	build_vtlapack build_rknave ${INSTALL_DIR}/lib/libdb.a

clear_db:
	-rm -f ${INSTALL_DIR}/lib/libdb.a
	-rm -r ${DEP_DIR}/db/build

${INSTALL_DIR}/lib/libdb.a:	${DEP_DIR}/db
	mkdir -p ${DEP_DIR}/db/build
	cd ${DEP_DIR}/db/build; cmake -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} -DBOOST_ROOT=${BOOST_ROOT} -DINCLUDE_BASE_DIR=${INSTALL_DIR} -DDBDC_SHARED_BUILD=OFF ../dbcmake
	cd ${DEP_DIR}/db/build; make install

${DEP_DIR}/db:
	echo GIT_USER=$(GIT_USER)
	cd ${DEP_DIR};git clone $(GIT_USER)@git-open.sarnoff.com:/scm/vision/db

##################################################################
build_feature_association:	${INSTALL_DIR}/lib/libFeatureAssociation.so

clear_feature_association:
	-rm -f ${INSTALL_DIR}/lib/libFeatureAssociation.so
	-rm -r ${DEP_DIR}/feature_association/build

${INSTALL_DIR}/lib/libFeatureAssociation.so:	${DEP_DIR}/feature_association
	mkdir -p ${DEP_DIR}/feature_association/build
	cd ${DEP_DIR}/feature_association/build;cmake -Dvtcmake_DIR=${INSTALL_DIR}/vtcmake -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} -DCMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu -DBoost_NO_SYSTEM_PATHS=true -DBOOST_ROOT=${BOOST_ROOT} -DOpenCV_DIR=${OpenCV_DIR} ..
	cd ${DEP_DIR}/feature_association/build; make install

${DEP_DIR}/feature_association:
	echo GIT_USER=$(GIT_USER)
	cd ${DEP_DIR};git clone $(GIT_USER)@git-open.sarnoff.com:/scm/vision/feature_association
	cd ${DEP_DIR}/feature_association;git checkout B-FA_FISH_TRACK_0707_2015

##################################################################
clear_fishtrack:	
	-rm -r ${PROJ_DIR}/build

build_fishtrack:	
	mkdir -p ${PROJ_DIR}/build
	touch ${PROJ_DIR}/BgModel/config.h
	cd ${PROJ_DIR}/build; cmake -Dvtcmake_DIR=${INSTALL_DIR}/vtcmake -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} -DBOOST_ROOT=${BOOST_ROOT} -DOpenCV_DIR=${OpenCV_DIR} ..
	cd ${PROJ_DIR}/build; make install


##################################################################
build_app:
	mkdir -p ${PROJ_DIR}/App/build
	cd ${PROJ_DIR}/App/build; cmake -DBOOST_ROOT=${BOOST_ROOT} -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} -DOpenCV_DIR=${OpenCV_DIR} ..
	cd ${PROJ_DIR}/App/build; make install

##################################################################
OPEVCV_VER=2.4.11
#build_opencv: ${DEP_DIR}/opencv-${OPEVCV_VER} ${DEP_DIR}/opencv_contrib build_opencv_obj
build_opencv: ${DEP_DIR}/opencv-${OPEVCV_VER}  ${INSTALL_DIR}/lib/libopencv_core.so

clear_opencv:
	-rm -r ${DEP_DIR}/opencv-${OPEVCV_VER}/build  ${INSTALL_DIR}/lib/libopencv_core.so

${INSTALL_DIR}/lib/libopencv_core.so:
	-mkdir ${DEP_DIR}/opencv-${OPEVCV_VER}/build
	cd ${DEP_DIR}/opencv-${OPEVCV_VER}/build;cmake -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} -DCUDA_GENERATION=Auto  ..
	cd ${DEP_DIR}/opencv-${OPEVCV_VER}/build;make install

	#cd ${DEP_DIR}/opencv_contrib; git checkout ${OPEVCV_VER}
	#-DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules

${DEP_DIR}/opencv-${OPEVCV_VER}:
	cd ${DEP_DIR}; wget https://github.com/Itseez/opencv/archive/${OPEVCV_VER}.zip
	cd ${DEP_DIR}; unzip ${OPEVCV_VER}.zip


${DEP_DIR}/opencv_contrib:
	cd ${DEP_DIR}; git clone https://github.com/Itseez/opencv_contrib.git

##################################################################
build_PythonModule:
	-mkdir -p ${PROJ_DIR}/PythonModule/build
	cd ${PROJ_DIR}/PythonModule/build; cmake -DBOOST_ROOT=${BOOST_ROOT} -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} -DBoost_NO_SYSTEM_PATHS=false -DOpenCV_DIR=${OpenCV_DIR} ..
	cd ${PROJ_DIR}/PythonModule/build; make install

clear_PythonModule:
	-rm -rf ${PROJ_DIR}/PythonModule/build

##################################################################
build_selective_search:	build_boost_numpy ${DEP_DIR}/selective_search_py
	mkdir -p ${DEP_DIR}/selective_search_py/build
	cd ${DEP_DIR}/selective_search_py/build; cmake -DBOOST_ROOT=${BOOST_ROOT} -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} ..
	cd ${DEP_DIR}/selective_search_py/build; make install

${DEP_DIR}/selective_search_py:
	echo GIT_USER=$(GIT_USER)
	cd ${DEP_DIR}; git clone $(GIT_USER)@git-open.sarnoff.com:/scm/vision/selective_search_py
#cd ${DEP_DIR}; git clone https://github.com/BradNeuberg/selective_search_py
#${DEP_DIR}/selective_search_py/segment:
#	cd ${DEP_DIR}/selective_search_py;	wget http://cs.brown.edu/~pff/segment/segment.zip; unzip segment.zip


##################################################################
build_boost_numpy:	${INSTALL_DIR}/lib/libboost_numpy.so

clear_boost_numpy:
	-rm -f ${INSTALL_DIR}/lib/libboost_numpy.so
	-rm	-r ${DEP_DIR}/Boost.NumPy/build

${INSTALL_DIR}/lib/libboost_numpy.so:	${DEP_DIR}/Boost.NumPy
	mkdir -p ${DEP_DIR}/Boost.NumPy/build
	cd ${DEP_DIR}/Boost.NumPy/build; cmake -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} -DBOOST_ROOT=${BOOST_ROOT} ..
	cd ${DEP_DIR}/Boost.NumPy/build; make install
	cd ${INSTALL_DIR}/lib64; cp * ${INSTALL_DIR}/lib

${DEP_DIR}/Boost.NumPy:
	cd ${DEP_DIR}; git clone https://github.com/ndarray/Boost.NumPy.git

##################################################################
build_caffe:	boost1.60_patch_caffe ${INSTALL_DIR}/lib/libcaffe.a

clear_caffe:
	-rm -f ${INSTALL_DIR}/lib/libcaffe.a
	-rm -r ${DEP_DIR}/caffe/build

${INSTALL_DIR}/lib/libcaffe.a:	${DEP_DIR}/caffe
	mkdir -p ${DEP_DIR}/caffe/build
	cd ${DEP_DIR}/caffe/build; cmake -DCMAKE_MODULE_PATH=${INSTALL_DIR} -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} -DOpenCV_DIR=${OpenCV_DIR} -DBOOST_ROOT=${BOOST_ROOT} -DBoost_NO_SYSTEM_PATHS=true -DBLAS=open ..
	cd ${DEP_DIR}/caffe/build; make install
#	cd ${DEP_DIR}/caffe/build; make runtest


${DEP_DIR}/caffe:
	cd ${DEP_DIR}; git clone https://github.com/BVLC/caffe.git

boost1.60_patch_caffe: ${INSTALL_DIR}/include/boost/config/compiler/gcc.hpp.patched
${INSTALL_DIR}/include/boost/config/compiler/gcc.hpp.patched:
	@echo local/include/boost/config/compiler/gcc.hpp
	@echo append '&& !defined(__CUDACC__)' at end of line 156
	@echo line 156: #if defined(_GLIBCXX_USE_FLOAT128) && !defined(__STRICT_ANSI__)
	sed -i 's/__STRICT_ANSI__)/__STRICT_ANSI__) \&\& !defined(__CUDACC__)/' ${INSTALL_DIR}/include/boost/config/compiler/gcc.hpp
	touch ${INSTALL_DIR}/include/boost/config/compiler/gcc.hpp.patched

##################################################################
build_openblas:	${INSTALL_DIR}/lib/libopenblas.so

${INSTALL_DIR}/lib/libopenblas.so:	${DEP_DIR}/OpenBLAS
		cd ${DEP_DIR}/OpenBLAS; make FC=gfortran PREFIX=${INSTALL_DIR}
		cd ${DEP_DIR}/OpenBLAS; make FC=gfortran PREFIX=${INSTALL_DIR} install

${DEP_DIR}/OpenBLAS:
	cd ${DEP_DIR}; git clone https://github.com/xianyi/OpenBLAS.git


##################################################################
${DEP_DIR} ${INSTALL_DIR}:
	mkdir -p $@

export:
	zip -r export.zip local/* python/* PythonApp/*
