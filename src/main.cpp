
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void haar(Mat img);

void haar(Mat img){

}

int main() {

    cout << "Hello" <<endl;

    Mat image = imread("../img/31.png");

    if(image.empty()) {
        cout << "无法加载图像!" << endl;
        return -1;
    }
    
    namedWindow("显示图像", WINDOW_AUTOSIZE);
    imshow("显示图像", image);
    
    waitKey(0);
    
    return 0;
}