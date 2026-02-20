#!/bin/bash


set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../../config"

# Detect OpenCV version
result=0 && pkg-config --list-all | grep -q opencv4 && result=1
if [ $result -eq 1 ]; then
    OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
    echo "Using OpenCV 4"
else
    OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
    echo "Using OpenCV 3"
fi

CXX=${CXX:-g++}
COMMON_FLAGS="-std=c++17 -O2 -I${SCRIPT_DIR} -I${CONFIG_DIR}"
COMMON_LIBS="-lvitis_ai_library-yolov3 -lvitis_ai_library-dpu_task \
    -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config \
    -lvitis_ai_library-math -lvart-util -lxir \
    -pthread -ljson-c -lglog \
    ${OPENCV_FLAGS} -lopencv_core -lopencv_videoio -lopencv_imgproc \
    -lopencv_imgcodecs -lopencv_highgui"

# Build standard applications
for src in detect_image.cpp detect_video.cpp detect_video_stream.cpp \
           test_video_perf.cpp test_accuracy.cpp; do
    if [ -f "$SCRIPT_DIR/$src" ]; then
        name=${src%.*}
        echo "Building ${name}..."
        $CXX $COMMON_FLAGS -o "$SCRIPT_DIR/$name" "$SCRIPT_DIR/$src" $COMMON_LIBS
    fi
done


if [ -f "$SCRIPT_DIR/detect_video_hdmi.cpp" ]; then
    echo "Building detect_video_hdmi (DRM)..."
    DRM_FLAGS=$(pkg-config --cflags --libs libdrm 2>/dev/null || echo "-I/usr/include/libdrm -I/usr/include/drm -ldrm")
    $CXX $COMMON_FLAGS $DRM_FLAGS -o "$SCRIPT_DIR/detect_video_hdmi" \
        "$SCRIPT_DIR/detect_video_hdmi.cpp" $COMMON_LIBS -ldrm
fi

echo ""
echo "=== Build complete ==="
echo "Executables:"
ls -la "$SCRIPT_DIR"/detect_image "$SCRIPT_DIR"/detect_video \
       "$SCRIPT_DIR"/detect_video_hdmi "$SCRIPT_DIR"/detect_video_stream \
       "$SCRIPT_DIR"/test_video_perf "$SCRIPT_DIR"/test_accuracy 2>/dev/null