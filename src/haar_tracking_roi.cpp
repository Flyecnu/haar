#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// === 配置参数 ===
const string CASCADE_PATH = "../haarcascade_drone4.xml";
const string IMAGE_FOLDER  = "../img/video_01";
const string OUTPUT_FOLDER = "../output_haar_roi120*120";
const int SEARCH_RADIUS = 60;  // 帧间局部搜索半径

vector<string> getSortedImagePaths(const string& folder_path) {
    vector<string> files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        files.push_back(entry.path().string());
    }
    sort(files.begin(), files.end());
    return files;
}

int main() {
    CascadeClassifier droneCascade;
    if (!droneCascade.load(CASCADE_PATH)) {
        cerr << "❌ Failed to load Haar classifier: " << CASCADE_PATH << endl;
        return -1;
    }

    fs::create_directories(OUTPUT_FOLDER);
    vector<string> image_paths = getSortedImagePaths(IMAGE_FOLDER);

    Point lastCenter(-1, -1);

    for (size_t i = 0; i < image_paths.size(); ++i) {
        string img_path = image_paths[i];
        Mat frame = imread(img_path, IMREAD_GRAYSCALE);
        if (frame.empty()) continue;

        int64 start = getTickCount();

        vector<Rect> detections;
        Mat display;
        cvtColor(frame, display, COLOR_GRAY2BGR);

        bool localSearchUsed = false;
        if (lastCenter.x >= 0 && lastCenter.y >= 0) {
            int x = max(lastCenter.x - SEARCH_RADIUS, 0);
            int y = max(lastCenter.y - SEARCH_RADIUS, 0);
            int w = min(2 * SEARCH_RADIUS, frame.cols - x);
            int h = min(2 * SEARCH_RADIUS, frame.rows - y);
            Rect roi(x, y, w, h);
            Mat roiImg = frame(roi);

            droneCascade.detectMultiScale(roiImg, detections, 1.1, 3, 0, Size(40, 40));

            // 坐标修正到全图坐标系
            for (auto& d : detections) {
                d.x += roi.x;
                d.y += roi.y;
            }

            localSearchUsed = true;
        }

        if (detections.empty()) {
            droneCascade.detectMultiScale(frame, detections, 1.1, 3, 0, Size(40, 40));
            localSearchUsed = false;
        }

        int64 end = getTickCount();
        double elapsed_ms = 1000.0 * (end - start) / getTickFrequency();

        if (!detections.empty()) {
            for (const auto& box : detections) {
                rectangle(display, box, Scalar(0, 255, 0), 2);
            }
            lastCenter = Point(
                detections[0].x + detections[0].width / 2,
                detections[0].y + detections[0].height / 2
            );
            cout << "✅ Frame " << i << " | Detections: " << detections.size()
                 << (localSearchUsed ? " | ROI ✅" : " | Full Img")
                 << " | Time: " << fixed << setprecision(2) << elapsed_ms << " ms" << endl;
        } else {
            lastCenter = Point(-1, -1);
            cout << "❌ Frame " << i << " | No detection | Time: " << fixed << setprecision(2) << elapsed_ms << " ms" << endl;
        }

        string out_path = OUTPUT_FOLDER + "/" + fs::path(img_path).stem().string() + "_haar_roi.jpg";
        imwrite(out_path, display);
        imshow("Haar Drone Detection with ROI", display);
        waitKey(1);
    }

    return 0;
}
