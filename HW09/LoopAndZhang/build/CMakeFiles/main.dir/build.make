# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.25.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.25.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/main.cpp.o: /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/src/main.cpp
CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/src/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/main.cpp.o -MF CMakeFiles/main.dir/src/main.cpp.o.d -o CMakeFiles/main.dir/src/main.cpp.o -c /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/src/main.cpp

CMakeFiles/main.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/src/main.cpp > CMakeFiles/main.dir/src/main.cpp.i

CMakeFiles/main.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/src/main.cpp -o CMakeFiles/main.dir/src/main.cpp.s

CMakeFiles/main.dir/src/util.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/util.cpp.o: /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/src/util.cpp
CMakeFiles/main.dir/src/util.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/src/util.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/util.cpp.o -MF CMakeFiles/main.dir/src/util.cpp.o.d -o CMakeFiles/main.dir/src/util.cpp.o -c /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/src/util.cpp

CMakeFiles/main.dir/src/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/util.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/src/util.cpp > CMakeFiles/main.dir/src/util.cpp.i

CMakeFiles/main.dir/src/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/util.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/src/util.cpp -o CMakeFiles/main.dir/src/util.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/main.cpp.o" \
"CMakeFiles/main.dir/src/util.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/src/main.cpp.o
main: CMakeFiles/main.dir/src/util.cpp.o
main: CMakeFiles/main.dir/build.make
main: /usr/local/lib/libopencv_gapi.4.6.0.dylib
main: /usr/local/lib/libopencv_stitching.4.6.0.dylib
main: /usr/local/lib/libopencv_alphamat.4.6.0.dylib
main: /usr/local/lib/libopencv_aruco.4.6.0.dylib
main: /usr/local/lib/libopencv_barcode.4.6.0.dylib
main: /usr/local/lib/libopencv_bgsegm.4.6.0.dylib
main: /usr/local/lib/libopencv_bioinspired.4.6.0.dylib
main: /usr/local/lib/libopencv_ccalib.4.6.0.dylib
main: /usr/local/lib/libopencv_dnn_objdetect.4.6.0.dylib
main: /usr/local/lib/libopencv_dnn_superres.4.6.0.dylib
main: /usr/local/lib/libopencv_dpm.4.6.0.dylib
main: /usr/local/lib/libopencv_face.4.6.0.dylib
main: /usr/local/lib/libopencv_freetype.4.6.0.dylib
main: /usr/local/lib/libopencv_fuzzy.4.6.0.dylib
main: /usr/local/lib/libopencv_hfs.4.6.0.dylib
main: /usr/local/lib/libopencv_img_hash.4.6.0.dylib
main: /usr/local/lib/libopencv_intensity_transform.4.6.0.dylib
main: /usr/local/lib/libopencv_line_descriptor.4.6.0.dylib
main: /usr/local/lib/libopencv_mcc.4.6.0.dylib
main: /usr/local/lib/libopencv_quality.4.6.0.dylib
main: /usr/local/lib/libopencv_rapid.4.6.0.dylib
main: /usr/local/lib/libopencv_reg.4.6.0.dylib
main: /usr/local/lib/libopencv_rgbd.4.6.0.dylib
main: /usr/local/lib/libopencv_saliency.4.6.0.dylib
main: /usr/local/lib/libopencv_sfm.4.6.0.dylib
main: /usr/local/lib/libopencv_stereo.4.6.0.dylib
main: /usr/local/lib/libopencv_structured_light.4.6.0.dylib
main: /usr/local/lib/libopencv_superres.4.6.0.dylib
main: /usr/local/lib/libopencv_surface_matching.4.6.0.dylib
main: /usr/local/lib/libopencv_tracking.4.6.0.dylib
main: /usr/local/lib/libopencv_videostab.4.6.0.dylib
main: /usr/local/lib/libopencv_viz.4.6.0.dylib
main: /usr/local/lib/libopencv_wechat_qrcode.4.6.0.dylib
main: /usr/local/lib/libopencv_xfeatures2d.4.6.0.dylib
main: /usr/local/lib/libopencv_xobjdetect.4.6.0.dylib
main: /usr/local/lib/libopencv_xphoto.4.6.0.dylib
main: /usr/local/lib/libopencv_shape.4.6.0.dylib
main: /usr/local/lib/libopencv_highgui.4.6.0.dylib
main: /usr/local/lib/libopencv_datasets.4.6.0.dylib
main: /usr/local/lib/libopencv_plot.4.6.0.dylib
main: /usr/local/lib/libopencv_text.4.6.0.dylib
main: /usr/local/lib/libopencv_ml.4.6.0.dylib
main: /usr/local/lib/libopencv_phase_unwrapping.4.6.0.dylib
main: /usr/local/lib/libopencv_optflow.4.6.0.dylib
main: /usr/local/lib/libopencv_ximgproc.4.6.0.dylib
main: /usr/local/lib/libopencv_video.4.6.0.dylib
main: /usr/local/lib/libopencv_videoio.4.6.0.dylib
main: /usr/local/lib/libopencv_imgcodecs.4.6.0.dylib
main: /usr/local/lib/libopencv_objdetect.4.6.0.dylib
main: /usr/local/lib/libopencv_calib3d.4.6.0.dylib
main: /usr/local/lib/libopencv_dnn.4.6.0.dylib
main: /usr/local/lib/libopencv_features2d.4.6.0.dylib
main: /usr/local/lib/libopencv_flann.4.6.0.dylib
main: /usr/local/lib/libopencv_photo.4.6.0.dylib
main: /usr/local/lib/libopencv_imgproc.4.6.0.dylib
main: /usr/local/lib/libopencv_core.4.6.0.dylib
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/build /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/build /Users/wang3450/Desktop/ECE661/HW09/LoopAndZhang/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

