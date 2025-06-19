#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// === 参数设置 ===
const string TEMPLATE_PATH = "../img/template_001.jpg";
const string FRAME_FOLDER  = "../img/video_01";
const string OUTPUT_FOLDER = "../output/6_2";
const string TIME_LOG_FILE = "../output/timelog.txt";
const double MATCH_THRESHOLD = 0.25;
const int SEARCH_RADIUS = 100;

// 木箱估计参数（你统计的平均值）
const double BOX_DX = 0.86;
const double BOX_DY = 187.57;
const double SCALE_W = 0.17;
const double SCALE_H = 0.53;

int main() {
    // 加载模板
    Mat templ = imread(TEMPLATE_PATH, IMREAD_GRAYSCALE);
    if (templ.empty()) {
        cerr << "❌ 无法读取模板图像：" << TEMPLATE_PATH << endl;
        return -1;
    }

    fs::create_directories(OUTPUT_FOLDER);
    ofstream timeLog(TIME_LOG_FILE);
    if (!timeLog.is_open()) {
        cerr << "❌ 无法打开时间日志文件" << endl;
        return -1;
    }

    // 收集帧
    vector<string> files;
    for (const auto& entry : fs::directory_iterator(FRAME_FOLDER)) {
        files.push_back(entry.path().string());
    }
    sort(files.begin(), files.end());

    Point prevCenter(-1, -1);
    int successCount = 0;
    int totalCount = 0;

    for (const auto& file : files) {
        Mat frame = imread(file, IMREAD_GRAYSCALE);
        if (frame.empty()) continue;

        auto start = chrono::high_resolution_clock::now();

        Rect searchRect(0, 0, frame.cols, frame.rows);
        if (prevCenter.x > 0 && prevCenter.y > 0) {
            int x = max(prevCenter.x - SEARCH_RADIUS, 0);
            int y = max(prevCenter.y - SEARCH_RADIUS, 0);
            int w = min(templ.cols + 2 * SEARCH_RADIUS, frame.cols - x);
            int h = min(templ.rows + 2 * SEARCH_RADIUS, frame.rows - y);
            searchRect = Rect(x, y, w, h);
        }

        Mat searchROI = frame(searchRect);
        Mat result;
        matchTemplate(searchROI, templ, result, TM_CCOEFF_NORMED);

        double maxVal;
        Point maxLoc;
        minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);

        Point matchCenter = Point(maxLoc.x + templ.cols / 2 + searchRect.x,
                                  maxLoc.y + templ.rows / 2 + searchRect.y);

        bool matched = maxVal >= MATCH_THRESHOLD;
        if (matched) {
            // 画无人机框
            Rect droneBox(matchCenter.x - templ.cols / 2, matchCenter.y - templ.rows / 2,
                          templ.cols, templ.rows);
            rectangle(frame, droneBox, Scalar(255), 2);

            // 木箱位置估计
            int box_cx = static_cast<int>(matchCenter.x + BOX_DX);
            int box_cy = static_cast<int>(matchCenter.y + BOX_DY);
            int box_w = static_cast<int>(templ.cols * SCALE_W);
            int box_h = static_cast<int>(templ.rows * SCALE_H);
            Rect boxROI(box_cx - box_w / 2, box_cy - box_h / 2, box_w, box_h);

            // 画木箱框（加边界保护）
            if (boxROI.x >= 0 && boxROI.y >= 0 &&
                boxROI.x + boxROI.width <= frame.cols &&
                boxROI.y + boxROI.height <= frame.rows) {
                rectangle(frame, boxROI, Scalar(128), 2);
            }

            // 更新前一帧中心
            prevCenter = matchCenter;
            successCount++;
        }

        totalCount++;
        auto end = chrono::high_resolution_clock::now();
        double elapsed_ms = chrono::duration<double, std::milli>(end - start).count();

        // 输出帧图像
        string out_name = OUTPUT_FOLDER + "/" + fs::path(file).filename().string();
        imwrite(out_name, frame);

        // 控制台输出
        cout << fs::path(file).filename() << " - 匹配: " << (matched ? "✔️" : "❌")
             << " - 得分: " << maxVal << " - 用时: " << elapsed_ms << " ms" << endl;

        // 写日志
        timeLog << fs::path(file).filename() << ", " << elapsed_ms << "\n";
    }

    timeLog.close();
    cout << "✅ 总帧数: " << totalCount
         << ", 检测成功: " << successCount
         << ", 成功率: " << (100.0 * successCount / totalCount) << "%" << endl;

    return 0;
}
