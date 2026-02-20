/*
 * detect_image.cpp
 */
#include "common.hpp"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_name> <image_path>" << std::endl;
        return -1;
    }

    std::string model_name = argv[1];
    std::string image_path = argv[2];

    // Load image
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "Error: could not load image " << image_path << std::endl;
        return -1;
    }

    // Create YOLO model — loads xmodel into DPU
    auto yolo = vitis::ai::YOLOv3::create(model_name, true);

    // Run DPU inference — identical execution path
    auto results = yolo->run(frame);

    // Draw detections — preserved coordinate logic
    common::draw_detections(frame, results);

    // Save result
    std::string output_path = "result_" + image_path.substr(image_path.find_last_of("/\\") + 1);
    cv::imwrite(output_path, frame);
    std::cout << "Result saved to " << output_path << std::endl;

    return 0;
}