#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// === 配置参数 ===
const string CASCADE_PATH = "../haarcascade_drone3.xml";
const string IMAGE_FOLDER  = "../img/video_01";
const string OUTPUT_FOLDER = "../output_haar_roi80*80";

vector<string> getSortedImagePaths(const string& folder_path) {
    vector<string> files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        files.push_back(entry.path().string());
    }
    sort(files.begin(), files.end());
    return files;
}

Rect getThresholdROI(const Mat& gray) {
    Mat thresh;
    threshold(gray, thresh, 80, 255, THRESH_BINARY);
    morphologyEx(thresh, thresh, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5, 5)));

    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Rect roi;
    int maxArea = 0;
    for (const auto& c : contours) {
        Rect r = boundingRect(c);
        int area = r.area();
        if (area > maxArea) {
            roi = r;
            maxArea = area;
        }
    }
    return roi;
}

int main() {
    CascadeClassifier droneCascade;
    if (!droneCascade.load(CASCADE_PATH)) {
        cerr << "❌ Failed to load Haar classifier: " << CASCADE_PATH << endl;
        return -1;
    }

    fs::create_directories(OUTPUT_FOLDER);
    vector<string> image_paths = getSortedImagePaths(IMAGE_FOLDER);

    for (size_t i = 0; i < image_paths.size(); ++i) {
        string img_path = image_paths[i];
        Mat gray = imread(img_path, IMREAD_GRAYSCALE);
        if (gray.empty()) continue;

        int64 start = getTickCount();

        // Step 1: 提取目标候选区域
        Rect roi = getThresholdROI(gray);
        if (roi.area() == 0) roi = Rect(0, 0, gray.cols, gray.rows);  // 兜底：全图

        Mat roiImg = gray(roi);
        vector<Rect> detections;
        droneCascade.detectMultiScale(roiImg, detections, 1.1, 5, 0, Size(80, 60), Size(160, 120));

        // 坐标修正到全图
        for (auto& d : detections) {
            d.x += roi.x;
            d.y += roi.y;
        }

        // Step 2: 可视化结果
        Mat display;
        cvtColor(gray, display, COLOR_GRAY2BGR);
        Rect bestBox;
        int maxArea = 0;

        for (const auto& box : detections) {
            int area = box.area();
            if (area > maxArea) {
                bestBox = box;
                maxArea = area;
            }
        }

        int64 end = getTickCount();
        double elapsed_ms = 1000.0 * (end - start) / getTickFrequency();

        if (maxArea > 0) {
            rectangle(display, bestBox, Scalar(0, 255, 0), 2);
            cout << "✅ Frame " << i << " | 1 Target | Time: "
                 << fixed << setprecision(2) << elapsed_ms << " ms" << endl;
        } else {
            cout << "❌ Frame " << i << " | No detection | Time: "
                 << fixed << setprecision(2) << elapsed_ms << " ms" << endl;
        }

        string out_path = OUTPUT_FOLDER + "/" + fs::path(img_path).stem().string() + "_haar_thresh.jpg";
        imwrite(out_path, display);
        imshow("Haar + Threshold Detection", display);
        waitKey(1);
    }

    return 0;
}
