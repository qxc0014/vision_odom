# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/esoman/c++code/slambook2/testcode/keypoint

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/esoman/c++code/slambook2/testcode/keypoint/build

# Include any dependencies generated for this target.
include CMakeFiles/keypoint.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/keypoint.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/keypoint.dir/flags.make

CMakeFiles/keypoint.dir/keypoint.cpp.o: CMakeFiles/keypoint.dir/flags.make
CMakeFiles/keypoint.dir/keypoint.cpp.o: ../keypoint.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/esoman/c++code/slambook2/testcode/keypoint/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/keypoint.dir/keypoint.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/keypoint.dir/keypoint.cpp.o -c /home/esoman/c++code/slambook2/testcode/keypoint/keypoint.cpp

CMakeFiles/keypoint.dir/keypoint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/keypoint.dir/keypoint.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/esoman/c++code/slambook2/testcode/keypoint/keypoint.cpp > CMakeFiles/keypoint.dir/keypoint.cpp.i

CMakeFiles/keypoint.dir/keypoint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/keypoint.dir/keypoint.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/esoman/c++code/slambook2/testcode/keypoint/keypoint.cpp -o CMakeFiles/keypoint.dir/keypoint.cpp.s

CMakeFiles/keypoint.dir/slamBase.cpp.o: CMakeFiles/keypoint.dir/flags.make
CMakeFiles/keypoint.dir/slamBase.cpp.o: ../slamBase.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/esoman/c++code/slambook2/testcode/keypoint/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/keypoint.dir/slamBase.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/keypoint.dir/slamBase.cpp.o -c /home/esoman/c++code/slambook2/testcode/keypoint/slamBase.cpp

CMakeFiles/keypoint.dir/slamBase.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/keypoint.dir/slamBase.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/esoman/c++code/slambook2/testcode/keypoint/slamBase.cpp > CMakeFiles/keypoint.dir/slamBase.cpp.i

CMakeFiles/keypoint.dir/slamBase.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/keypoint.dir/slamBase.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/esoman/c++code/slambook2/testcode/keypoint/slamBase.cpp -o CMakeFiles/keypoint.dir/slamBase.cpp.s

CMakeFiles/keypoint.dir/calTwist.cpp.o: CMakeFiles/keypoint.dir/flags.make
CMakeFiles/keypoint.dir/calTwist.cpp.o: ../calTwist.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/esoman/c++code/slambook2/testcode/keypoint/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/keypoint.dir/calTwist.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/keypoint.dir/calTwist.cpp.o -c /home/esoman/c++code/slambook2/testcode/keypoint/calTwist.cpp

CMakeFiles/keypoint.dir/calTwist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/keypoint.dir/calTwist.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/esoman/c++code/slambook2/testcode/keypoint/calTwist.cpp > CMakeFiles/keypoint.dir/calTwist.cpp.i

CMakeFiles/keypoint.dir/calTwist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/keypoint.dir/calTwist.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/esoman/c++code/slambook2/testcode/keypoint/calTwist.cpp -o CMakeFiles/keypoint.dir/calTwist.cpp.s

CMakeFiles/keypoint.dir/optimize.cpp.o: CMakeFiles/keypoint.dir/flags.make
CMakeFiles/keypoint.dir/optimize.cpp.o: ../optimize.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/esoman/c++code/slambook2/testcode/keypoint/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/keypoint.dir/optimize.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/keypoint.dir/optimize.cpp.o -c /home/esoman/c++code/slambook2/testcode/keypoint/optimize.cpp

CMakeFiles/keypoint.dir/optimize.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/keypoint.dir/optimize.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/esoman/c++code/slambook2/testcode/keypoint/optimize.cpp > CMakeFiles/keypoint.dir/optimize.cpp.i

CMakeFiles/keypoint.dir/optimize.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/keypoint.dir/optimize.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/esoman/c++code/slambook2/testcode/keypoint/optimize.cpp -o CMakeFiles/keypoint.dir/optimize.cpp.s

CMakeFiles/keypoint.dir/showimage.cpp.o: CMakeFiles/keypoint.dir/flags.make
CMakeFiles/keypoint.dir/showimage.cpp.o: ../showimage.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/esoman/c++code/slambook2/testcode/keypoint/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/keypoint.dir/showimage.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/keypoint.dir/showimage.cpp.o -c /home/esoman/c++code/slambook2/testcode/keypoint/showimage.cpp

CMakeFiles/keypoint.dir/showimage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/keypoint.dir/showimage.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/esoman/c++code/slambook2/testcode/keypoint/showimage.cpp > CMakeFiles/keypoint.dir/showimage.cpp.i

CMakeFiles/keypoint.dir/showimage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/keypoint.dir/showimage.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/esoman/c++code/slambook2/testcode/keypoint/showimage.cpp -o CMakeFiles/keypoint.dir/showimage.cpp.s

# Object files for target keypoint
keypoint_OBJECTS = \
"CMakeFiles/keypoint.dir/keypoint.cpp.o" \
"CMakeFiles/keypoint.dir/slamBase.cpp.o" \
"CMakeFiles/keypoint.dir/calTwist.cpp.o" \
"CMakeFiles/keypoint.dir/optimize.cpp.o" \
"CMakeFiles/keypoint.dir/showimage.cpp.o"

# External object files for target keypoint
keypoint_EXTERNAL_OBJECTS =

keypoint: CMakeFiles/keypoint.dir/keypoint.cpp.o
keypoint: CMakeFiles/keypoint.dir/slamBase.cpp.o
keypoint: CMakeFiles/keypoint.dir/calTwist.cpp.o
keypoint: CMakeFiles/keypoint.dir/optimize.cpp.o
keypoint: CMakeFiles/keypoint.dir/showimage.cpp.o
keypoint: CMakeFiles/keypoint.dir/build.make
keypoint: /usr/local/lib/libopencv_superres.so.3.4.8
keypoint: /usr/local/lib/libopencv_dnn.so.3.4.8
keypoint: /usr/local/lib/libopencv_highgui.so.3.4.8
keypoint: /usr/local/lib/libopencv_objdetect.so.3.4.8
keypoint: /usr/local/lib/libopencv_stitching.so.3.4.8
keypoint: /usr/local/lib/libopencv_videostab.so.3.4.8
keypoint: /usr/local/lib/libopencv_ml.so.3.4.8
keypoint: /usr/local/lib/libopencv_calib3d.so.3.4.8
keypoint: /usr/local/lib/libopencv_shape.so.3.4.8
keypoint: /usr/local/lib/libpcl_surface.so
keypoint: /usr/local/lib/libpcl_keypoints.so
keypoint: /usr/local/lib/libpcl_tracking.so
keypoint: /usr/local/lib/libpcl_recognition.so
keypoint: /usr/local/lib/libpcl_stereo.so
keypoint: /usr/local/lib/libpcl_outofcore.so
keypoint: /usr/local/lib/libpcl_people.so
keypoint: /usr/lib/x86_64-linux-gnu/libboost_system.so
keypoint: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
keypoint: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
keypoint: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
keypoint: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
keypoint: /usr/lib/x86_64-linux-gnu/libboost_regex.so
keypoint: /usr/lib/x86_64-linux-gnu/libqhull.so
keypoint: /usr/lib/libOpenNI.so
keypoint: /usr/lib/libOpenNI2.so
keypoint: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
keypoint: /usr/local/lib/libpangolin.so
keypoint: /usr/local/lib/libceres.a
keypoint: /usr/local/lib/libopencv_videoio.so.3.4.8
keypoint: /usr/local/lib/libopencv_photo.so.3.4.8
keypoint: /usr/local/lib/libopencv_imgcodecs.so.3.4.8
keypoint: /usr/local/lib/libopencv_features2d.so.3.4.8
keypoint: /usr/local/lib/libopencv_flann.so.3.4.8
keypoint: /usr/local/lib/libopencv_video.so.3.4.8
keypoint: /usr/local/lib/libopencv_imgproc.so.3.4.8
keypoint: /usr/local/lib/libopencv_core.so.3.4.8
keypoint: /usr/local/lib/libpcl_registration.so
keypoint: /usr/local/lib/libpcl_segmentation.so
keypoint: /usr/local/lib/libpcl_features.so
keypoint: /usr/local/lib/libpcl_filters.so
keypoint: /usr/local/lib/libpcl_sample_consensus.so
keypoint: /usr/local/lib/libvtkChartsCore-7.1.so.1
keypoint: /usr/local/lib/libvtkInfovisCore-7.1.so.1
keypoint: /usr/local/lib/libvtkIOGeometry-7.1.so.1
keypoint: /usr/local/lib/libvtkIOLegacy-7.1.so.1
keypoint: /usr/local/lib/libvtkIOPLY-7.1.so.1
keypoint: /usr/local/lib/libvtkRenderingLOD-7.1.so.1
keypoint: /usr/local/lib/libvtkViewsContext2D-7.1.so.1
keypoint: /usr/local/lib/libvtkViewsCore-7.1.so.1
keypoint: /usr/local/lib/libvtkInteractionWidgets-7.1.so.1
keypoint: /usr/local/lib/libvtkFiltersModeling-7.1.so.1
keypoint: /usr/local/lib/libvtkInteractionStyle-7.1.so.1
keypoint: /usr/local/lib/libvtkFiltersExtraction-7.1.so.1
keypoint: /usr/local/lib/libvtkFiltersStatistics-7.1.so.1
keypoint: /usr/local/lib/libvtkImagingFourier-7.1.so.1
keypoint: /usr/local/lib/libvtkalglib-7.1.so.1
keypoint: /usr/local/lib/libvtkFiltersHybrid-7.1.so.1
keypoint: /usr/local/lib/libvtkImagingGeneral-7.1.so.1
keypoint: /usr/local/lib/libvtkImagingSources-7.1.so.1
keypoint: /usr/local/lib/libvtkImagingHybrid-7.1.so.1
keypoint: /usr/local/lib/libvtkRenderingAnnotation-7.1.so.1
keypoint: /usr/local/lib/libvtkImagingColor-7.1.so.1
keypoint: /usr/local/lib/libvtkRenderingVolume-7.1.so.1
keypoint: /usr/local/lib/libvtkIOXML-7.1.so.1
keypoint: /usr/local/lib/libvtkIOXMLParser-7.1.so.1
keypoint: /usr/local/lib/libvtkIOCore-7.1.so.1
keypoint: /usr/local/lib/libvtkexpat-7.1.so.1
keypoint: /usr/local/lib/libvtkRenderingContextOpenGL2-7.1.so.1
keypoint: /usr/local/lib/libvtkRenderingContext2D-7.1.so.1
keypoint: /usr/local/lib/libvtkRenderingFreeType-7.1.so.1
keypoint: /usr/local/lib/libvtkfreetype-7.1.so.1
keypoint: /usr/local/lib/libvtkRenderingOpenGL2-7.1.so.1
keypoint: /usr/local/lib/libvtkImagingCore-7.1.so.1
keypoint: /usr/local/lib/libvtkRenderingCore-7.1.so.1
keypoint: /usr/local/lib/libvtkCommonColor-7.1.so.1
keypoint: /usr/local/lib/libvtkFiltersGeometry-7.1.so.1
keypoint: /usr/local/lib/libvtkFiltersSources-7.1.so.1
keypoint: /usr/local/lib/libvtkFiltersGeneral-7.1.so.1
keypoint: /usr/local/lib/libvtkCommonComputationalGeometry-7.1.so.1
keypoint: /usr/local/lib/libvtkFiltersCore-7.1.so.1
keypoint: /usr/local/lib/libvtkIOImage-7.1.so.1
keypoint: /usr/local/lib/libvtkCommonExecutionModel-7.1.so.1
keypoint: /usr/local/lib/libvtkCommonDataModel-7.1.so.1
keypoint: /usr/local/lib/libvtkCommonTransforms-7.1.so.1
keypoint: /usr/local/lib/libvtkCommonMisc-7.1.so.1
keypoint: /usr/local/lib/libvtkCommonMath-7.1.so.1
keypoint: /usr/local/lib/libvtkCommonSystem-7.1.so.1
keypoint: /usr/local/lib/libvtkCommonCore-7.1.so.1
keypoint: /usr/local/lib/libvtksys-7.1.so.1
keypoint: /usr/local/lib/libvtkDICOMParser-7.1.so.1
keypoint: /usr/local/lib/libvtkmetaio-7.1.so.1
keypoint: /usr/local/lib/libvtkpng-7.1.so.1
keypoint: /usr/local/lib/libvtktiff-7.1.so.1
keypoint: /usr/local/lib/libvtkzlib-7.1.so.1
keypoint: /usr/local/lib/libvtkjpeg-7.1.so.1
keypoint: /usr/lib/x86_64-linux-gnu/libm.so
keypoint: /usr/lib/x86_64-linux-gnu/libXt.so
keypoint: /usr/local/lib/libvtkglew-7.1.so.1
keypoint: /usr/local/lib/libpcl_ml.so
keypoint: /usr/local/lib/libpcl_visualization.so
keypoint: /usr/local/lib/libpcl_search.so
keypoint: /usr/local/lib/libpcl_kdtree.so
keypoint: /usr/local/lib/libpcl_io.so
keypoint: /usr/local/lib/libpcl_octree.so
keypoint: /usr/local/lib/libpcl_common.so
keypoint: /usr/lib/x86_64-linux-gnu/libGL.so
keypoint: /usr/lib/x86_64-linux-gnu/libGLU.so
keypoint: /usr/lib/x86_64-linux-gnu/libGLEW.so
keypoint: /usr/lib/x86_64-linux-gnu/libEGL.so
keypoint: /usr/lib/x86_64-linux-gnu/libwayland-client.so
keypoint: /usr/lib/x86_64-linux-gnu/libwayland-egl.so
keypoint: /usr/lib/x86_64-linux-gnu/libwayland-cursor.so
keypoint: /usr/lib/x86_64-linux-gnu/libSM.so
keypoint: /usr/lib/x86_64-linux-gnu/libICE.so
keypoint: /usr/lib/x86_64-linux-gnu/libX11.so
keypoint: /usr/lib/x86_64-linux-gnu/libXext.so
keypoint: /usr/lib/x86_64-linux-gnu/libdc1394.so
keypoint: /opt/ros/kinetic/lib/librealsense.so
keypoint: /usr/lib/libOpenNI.so
keypoint: /usr/lib/libOpenNI2.so
keypoint: /usr/lib/x86_64-linux-gnu/libpng.so
keypoint: /usr/lib/x86_64-linux-gnu/libz.so
keypoint: /usr/lib/x86_64-linux-gnu/libjpeg.so
keypoint: /usr/lib/x86_64-linux-gnu/libtiff.so
keypoint: /usr/lib/x86_64-linux-gnu/libIlmImf.so
keypoint: /usr/lib/x86_64-linux-gnu/liblz4.so
keypoint: /usr/lib/x86_64-linux-gnu/libglog.so
keypoint: /usr/lib/x86_64-linux-gnu/libgflags.so
keypoint: /usr/lib/x86_64-linux-gnu/libspqr.so
keypoint: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
keypoint: /usr/lib/x86_64-linux-gnu/libtbb.so
keypoint: /usr/lib/x86_64-linux-gnu/libcholmod.so
keypoint: /usr/lib/x86_64-linux-gnu/libccolamd.so
keypoint: /usr/lib/x86_64-linux-gnu/libcamd.so
keypoint: /usr/lib/x86_64-linux-gnu/libcolamd.so
keypoint: /usr/lib/x86_64-linux-gnu/libamd.so
keypoint: /usr/lib/liblapack.so
keypoint: /usr/lib/libf77blas.so
keypoint: /usr/lib/libatlas.so
keypoint: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
keypoint: /usr/lib/x86_64-linux-gnu/librt.so
keypoint: /usr/lib/x86_64-linux-gnu/libmetis.so
keypoint: /usr/lib/x86_64-linux-gnu/libcxsparse.so
keypoint: /usr/lib/liblapack.so
keypoint: /usr/lib/libf77blas.so
keypoint: /usr/lib/libatlas.so
keypoint: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
keypoint: /usr/lib/x86_64-linux-gnu/librt.so
keypoint: /usr/lib/x86_64-linux-gnu/libmetis.so
keypoint: /usr/lib/x86_64-linux-gnu/libcxsparse.so
keypoint: CMakeFiles/keypoint.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/esoman/c++code/slambook2/testcode/keypoint/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable keypoint"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/keypoint.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/keypoint.dir/build: keypoint

.PHONY : CMakeFiles/keypoint.dir/build

CMakeFiles/keypoint.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/keypoint.dir/cmake_clean.cmake
.PHONY : CMakeFiles/keypoint.dir/clean

CMakeFiles/keypoint.dir/depend:
	cd /home/esoman/c++code/slambook2/testcode/keypoint/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/esoman/c++code/slambook2/testcode/keypoint /home/esoman/c++code/slambook2/testcode/keypoint /home/esoman/c++code/slambook2/testcode/keypoint/build /home/esoman/c++code/slambook2/testcode/keypoint/build /home/esoman/c++code/slambook2/testcode/keypoint/build/CMakeFiles/keypoint.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/keypoint.dir/depend

