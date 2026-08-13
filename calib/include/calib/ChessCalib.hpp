#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class ChessCalib {
public:
    ChessCalib(int col, int row, const std::string& img_dir, const std::string& out_dir = "valid_calib_images");
    int processFolder(); // 处理整个图片文件夹

private:
    int col;          // 标定板列数
    int row;          // 标定板行数
    int saved_count;  // 有效图像计数
    std::string img_dir;  // 输入图片文件夹路径
    std::string out_dir;  // 有效图像输出文件夹

    // 单张图片处理（复用原核心逻辑）
    bool processSingleImage(const cv::Mat& frame, const std::string& save_path);
};

