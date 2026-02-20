/*
 * common.hpp
 */
#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vitis/ai/yolov3.hpp>

namespace common {

// ──────────────────────────────────────────────
// Camera
// ──────────────────────────────────────────────

/**
 * Open camera with GStreamer pipeline for See3CAM_CU30.
 * Returns true if camera opened successfully.
 */
inline bool open_camera(cv::VideoCapture& cap, const std::string& device = "/dev/video0") {
    std::string pipeline =
        "v4l2src device=" + device + " ! video/x-raw, width=640, height=480, "
        "framerate=30/1 ! videoconvert ! appsink";
    cap.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open camera " << device << std::endl;
        return false;
    }
    return true;
}

/**
 * Open camera with explicit BGR format conversion.
 */
inline bool open_camera_bgr(cv::VideoCapture& cap, const std::string& device = "/dev/video0") {
    std::string pipeline =
        "v4l2src device=" + device + " ! video/x-raw, width=640, height=480, "
        "framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink";
    cap.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open camera " << device << std::endl;
        return false;
    }
    return true;
}

// ──────────────────────────────────────────────
// Drawing
// ──────────────────────────────────────────────

/**
 * Draw YOLO detection results on frame.
 */
inline void draw_detections(cv::Mat& frame,
                            const vitis::ai::YOLOv3Result& results) {
    static const cv::Scalar colors[] = {
        cv::Scalar(0, 255, 0),     // 2-wheelers
        cv::Scalar(255, 128, 0),   // auto
        cv::Scalar(0, 0, 255),     // bus
        cv::Scalar(255, 255, 0),   // car
        cv::Scalar(255, 0, 255),   // pedestrian
        cv::Scalar(0, 255, 255),   // truck
    };
    static const std::string class_names[] = {
        "2-wheelers", "auto", "bus", "car", "pedestrian", "truck"
    };
    static const int num_classes = 6;

    for (const auto& box : results.bboxes) {
        int label = box.label;
        float xmin = box.x * frame.cols;
        float ymin = box.y * frame.rows;
        float xmax = xmin + box.width * frame.cols;
        float ymax = ymin + box.height * frame.rows;

      
        xmin = std::max(0.0f, xmin);
        ymin = std::max(0.0f, ymin);
        xmax = std::min((float)frame.cols, xmax);
        ymax = std::min((float)frame.rows, ymax);

        cv::Scalar color = colors[label % num_classes];
        std::string label_text = class_names[label % num_classes] +
                                 " " + std::to_string(int(box.score * 100)) + "%";

        cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), color, 2);

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX,
                                              0.5, 1, &baseline);
        cv::rectangle(frame,
                      cv::Point(xmin, ymin - text_size.height - 6),
                      cv::Point(xmin + text_size.width + 4, ymin),
                      color, -1);
        cv::putText(frame, label_text, cv::Point(xmin + 2, ymin - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// ──────────────────────────────────────────────
// FPS Measurement
// ──────────────────────────────────────────────

/**
 * Simple FPS counter. Call tick() every frame.
 */
class FpsCounter {
public:
    FpsCounter() : frame_count_(0), fps_(0.0) {
        t_start_ = (double)cv::getTickCount();
    }

    /** Call once per frame. Returns true when FPS was recalculated. */
    bool tick() {
        frame_count_++;
        double t_now = (double)cv::getTickCount();
        double elapsed = (t_now - t_start_) / cv::getTickFrequency();
        if (elapsed >= 1.0) {
            fps_ = frame_count_ / elapsed;
            frame_count_ = 0;
            t_start_ = t_now;
            return true;
        }
        return false;
    }

    double fps() const { return fps_; }

private:
    int frame_count_;
    double fps_;
    double t_start_;
};

// ──────────────────────────────────────────────
// Argument parsing
// ──────────────────────────────────────────────

/**
 * Validate model name argument.
 */
inline bool parse_model_arg(int argc, char* argv[], std::string& model_name) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_name>" << std::endl;
        std::cerr << "  e.g.: " << argv[0] << " yolov3_quant" << std::endl;
        return false;
    }
    model_name = argv[1];
    return true;
}

} // namespace common

#endif // COMMON_HPP