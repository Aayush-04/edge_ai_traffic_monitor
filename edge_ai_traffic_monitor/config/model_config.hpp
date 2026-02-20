/*
 * model_config.hpp

 */
#ifndef MODEL_CONFIG_HPP
#define MODEL_CONFIG_HPP

#include <string>
#include <vector>

namespace config {

// Detection classes 
static const std::vector<std::string> CLASS_NAMES = {
    "2-wheelers", "auto", "bus", "car", "pedestrian", "truck"
};
static const int NUM_CLASSES = 6;

// Detection thresholds
static const float CONF_THRESHOLD = 0.3f;
static const float NMS_THRESHOLD  = 0.45f;

// Camera pipeline for See3CAM_CU30 (UYVY format)
static const std::string CAMERA_PIPELINE =
    "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, "
    "framerate=30/1 ! videoconvert ! appsink";

static const std::string CAMERA_PIPELINE_BGR =
    "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, "
    "framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink";

// Colors for bounding boxes (BGR format for OpenCV)
static const cv::Scalar BOX_COLORS[] = {
    cv::Scalar(0, 255, 0),     // green      - 2-wheelers
    cv::Scalar(255, 128, 0),   // orange     - auto
    cv::Scalar(0, 0, 255),     // red        - bus
    cv::Scalar(255, 255, 0),   // cyan       - car
    cv::Scalar(255, 0, 255),   // magenta    - pedestrian
    cv::Scalar(0, 255, 255),   // yellow     - truck
};

// Default model name for Vitis AI Library
static const std::string DEFAULT_MODEL_NAME = "yolov3_quant";

} // namespace config

#endif // MODEL_CONFIG_HPP