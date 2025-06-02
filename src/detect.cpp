#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

const string IMAGE_FOLDER = "../img/video_01";
const string OUTPUT_FOLDER = "../output_contour";

vector<string> getSortedImagePaths(const string& folder_path) {
    vector<string> files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        files.push_back(entry.path().string());
    }
    sort(files.begin(), files.end());
    return files;
}

int main() {
    fs::create_directories(OUTPUT_FOLDER);
    vector<string> image_paths = getSortedImagePaths(IMAGE_FOLDER);

    Point lastDetected(-1, -1);
    KalmanFilter KF(4, 2, 0);
    KF.transitionMatrix = (Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);
    KF.measurementMatrix = Mat::eye(2, 4, CV_32F);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));
    setIdentity(KF.errorCovPost, Scalar::all(1));

    for (size_t i = 0; i < image_paths.size(); ++i) {
        string img_path = image_paths[i];
        Mat frame = imread(img_path, IMREAD_GRAYSCALE);
        if (frame.empty()) continue;

        int64 start = getTickCount();

        Mat blurred, edges;
        GaussianBlur(frame, blurred, Size(5, 5), 1.5);

        Rect roiRect(0, 0, frame.cols, frame.rows);
        if (lastDetected.x >= 0) {
            int r = 40;
            roiRect = Rect(max(lastDetected.x - r, 0), max(lastDetected.y - r, 0),
                          min(2 * r, frame.cols - max(lastDetected.x - r, 0)),
                          min(2 * r, frame.rows - max(lastDetected.y - r, 0)));
        }

        Mat roi = blurred(roiRect);
        Canny(roi, edges, 50, 150);

        vector<vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat display;
        cvtColor(frame, display, COLOR_GRAY2BGR);
        Rect bestBox;
        double bestScore = 0;

        for (const auto& cnt : contours) {
            vector<Point> approx;
            approxPolyDP(cnt, approx, arcLength(cnt, true)*0.02, true);
            if (approx.size() == 4 && isContourConvex(approx)) {
                Rect box = boundingRect(approx);
                box.x += roiRect.x;
                box.y += roiRect.y;
                float aspectRatio = (float)box.width / box.height;
                float area = box.area();
                if (aspectRatio > 0.8 && aspectRatio < 1.2 && area > 3000 && area < 10000) {
                    if (area > bestScore) {
                        bestScore = area;
                        bestBox = box;
                    }
                }
            }
        }

        Point detectedCenter(-1, -1);
        if (bestScore > 0) {
            rectangle(display, bestBox, Scalar(0, 255, 0), 2);
            detectedCenter = Point(bestBox.x + bestBox.width / 2, bestBox.y + bestBox.height / 2);
            Mat meas = (Mat_<float>(2,1) << detectedCenter.x, detectedCenter.y);
            KF.correct(meas);
            lastDetected = detectedCenter;
            cout << "✅ Frame " << i << " | Box at (" << bestBox.x << ", " << bestBox.y << ")"
                 << " | Score: " << bestScore;
        } else {
            Mat prediction = KF.predict();
            lastDetected = Point(prediction.at<float>(0), prediction.at<float>(1));
            rectangle(display, Rect(lastDetected.x - 40, lastDetected.y - 40, 80, 80), Scalar(0, 255, 255), 2);
            cout << "❌ Frame " << i << " | Prediction used at (" << lastDetected.x << ", " << lastDetected.y << ")";
        }

        int64 end = getTickCount();
        double elapsed_ms = 1000.0 * (end - start) / getTickFrequency();
        cout << fixed << setprecision(2) << " | Time: " << elapsed_ms << " ms" << endl;

        string out_path = OUTPUT_FOLDER + "/" + fs::path(img_path).stem().string() + "_contour.jpg";
        imwrite(out_path, display);
        imshow("Contour Detection with Kalman", display);
        waitKey(1);
    }

    return 0;
}