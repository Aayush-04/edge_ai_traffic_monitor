/*
 * test_accuracy.cpp
 * mAP accuracy evaluation on a validation dataset.
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/yolov3.hpp>
#include "common.hpp"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

struct GtBox {
    int label;
    float cx, cy, w, h;
};

static vector<GtBox> load_labels(const string& path) {
    vector<GtBox> boxes;
    ifstream f(path);
    if (!f.is_open()) return boxes;
    string line;
    while (getline(f, line)) {
        istringstream iss(line);
        GtBox b;
        if (iss >> b.label >> b.cx >> b.cy >> b.w >> b.h) {
            boxes.push_back(b);
        }
    }
    return boxes;
}

static float compute_iou(float x1a, float y1a, float x2a, float y2a,
                         float x1b, float y1b, float x2b, float y2b) {
    float xx1 = max(x1a, x1b), yy1 = max(y1a, y1b);
    float xx2 = min(x2a, x2b), yy2 = min(y2a, y2b);
    float inter = max(0.0f, xx2 - xx1) * max(0.0f, yy2 - yy1);
    float area_a = (x2a - x1a) * (y2a - y1a);
    float area_b = (x2b - x1b) * (y2b - y1b);
    float union_area = area_a + area_b - inter;
    return (union_area > 0) ? inter / union_area : 0.0f;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <model_name> <dataset_images_dir>" << endl;
        return -1;
    }

    string model_name = argv[1];
    string images_dir = argv[2];

    // Derive labels directory from images directory
    string labels_dir = images_dir;
    size_t pos = labels_dir.find("images");
    if (pos != string::npos) labels_dir.replace(pos, 6, "labels");

    auto yolo = vitis::ai::YOLOv3::create(model_name, true);

    int total_gt = 0, total_det = 0, true_pos = 0;
    int image_count = 0;
    float iou_threshold = 0.5f;

    for (const auto& entry : fs::directory_iterator(images_dir)) {
        if (entry.path().extension() != ".jpg" && entry.path().extension() != ".png") continue;

        Mat frame = imread(entry.path().string());
        if (frame.empty()) continue;

        auto results = yolo->run(frame);
        image_count++;

        // Load ground truth
        string label_file = labels_dir + "/" + entry.path().stem().string() + ".txt";
        vector<GtBox> gt_boxes = load_labels(label_file);
        total_gt += gt_boxes.size();
        total_det += results.bboxes.size();

        // Match detections to ground truth
        vector<bool> gt_matched(gt_boxes.size(), false);
        for (const auto& det : results.bboxes) {
            float dx1 = det.x * frame.cols;
            float dy1 = det.y * frame.rows;
            float dx2 = dx1 + det.width * frame.cols;
            float dy2 = dy1 + det.height * frame.rows;

            float best_iou = 0;
            int best_idx = -1;

            for (size_t g = 0; g < gt_boxes.size(); g++) {
                if (gt_matched[g]) continue;
                if (gt_boxes[g].label != det.label) continue;

                float gx1 = (gt_boxes[g].cx - gt_boxes[g].w / 2) * frame.cols;
                float gy1 = (gt_boxes[g].cy - gt_boxes[g].h / 2) * frame.rows;
                float gx2 = (gt_boxes[g].cx + gt_boxes[g].w / 2) * frame.cols;
                float gy2 = (gt_boxes[g].cy + gt_boxes[g].h / 2) * frame.rows;

                float iou = compute_iou(dx1, dy1, dx2, dy2, gx1, gy1, gx2, gy2);
                if (iou > best_iou) { best_iou = iou; best_idx = g; }
            }

            if (best_iou >= iou_threshold && best_idx >= 0) {
                true_pos++;
                gt_matched[best_idx] = true;
            }
        }

        if (image_count % 50 == 0) {
            cout << "Processed " << image_count << " images..." << endl;
        }
    }

    float precision = (total_det > 0) ? (float)true_pos / total_det : 0;
    float recall = (total_gt > 0) ? (float)true_pos / total_gt : 0;
    float f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;

    cout << "\n=== Accuracy Results ===" << endl;
    cout << "Images evaluated:  " << image_count << endl;
    cout << "Ground truth boxes:" << total_gt << endl;
    cout << "Detected boxes:    " << total_det << endl;
    cout << "True positives:    " << true_pos << endl;
    cout << "Precision:         " << precision << endl;
    cout << "Recall:            " << recall << endl;
    cout << "F1 Score:          " << f1 << endl;
    cout << "IoU threshold:     " << iou_threshold << endl;

    return 0;
}