#include "calib/ChessCalib.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>

// 构造函数：新增输入/输出文件夹参数
ChessCalib::ChessCalib(int col, int row, const std::string& img_dir, const std::string& out_dir) {
    this->col = col;
    this->row = row;
    this->saved_count = 0;
    this->img_dir = img_dir;
    this->out_dir = out_dir;

    // 创建输出目录（不存在则创建）
    mkdir(out_dir.c_str(), 0777);
}

// 单张图片处理：保留标定板检测，自动保存有效图像
bool ChessCalib::processSingleImage(const cv::Mat& frame, const std::string& save_path) {
    cv::Size patternSize(col, row);
    std::vector<cv::Point2f> centers;

    // 检测对称圆点标定板
    bool found = cv::findCirclesGrid(
        frame,
        patternSize,
        centers,
        cv::CALIB_CB_SYMMETRIC_GRID
    );

    if (found) {
        // 保存有效图像
        if (cv::imwrite(save_path, frame)) {
            saved_count++;
            std::cout << "✅ 有效图像: " << save_path << std::endl;
            return true;
        } else {
            std::cerr << "❌ 保存失败: " << save_path << std::endl;
            return false;
        }
    } else {
        std::cout << "⚠️ 无效图像（未检测到标定板）: " << save_path << std::endl;
        return false;
    }
}

// 遍历文件夹处理所有图片
int ChessCalib::processFolder() {
    DIR* dir = opendir(img_dir.c_str());
    if (!dir) {
        std::cerr << "❌ 无法打开文件夹: " << img_dir << std::endl;
        return -1;
    }

    dirent* entry;
    std::vector<std::string> img_files;
    // 筛选图片文件（支持jpg/png/jpeg）
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.find(".jpg") != std::string::npos ||
            filename.find(".png") != std::string::npos ||
            filename.find(".jpeg") != std::string::npos) {
            img_files.push_back(img_dir + "/" + filename);
        }
    }
    closedir(dir);

    if (img_files.empty()) {
        std::cerr << "❌ 文件夹中未找到图片文件" << std::endl;
        return -1;
    }

    std::cout << "📂 找到 " << img_files.size() << " 张图片，开始检测标定板..." << std::endl;

    // 处理每张图片
    for (size_t i = 0; i < img_files.size(); i++) {
        cv::Mat frame = cv::imread(img_files[i]);
        if (frame.empty()) {
            std::cerr << "❌ 无法读取图片: " << img_files[i] << std::endl;
            continue;
        }

        // 生成输出文件名：01.jpg, 02.jpg...
        char save_path[256];
        snprintf(save_path, sizeof(save_path), "%s/%02d.jpg", out_dir.c_str(), saved_count + 1);
        processSingleImage(frame, save_path);

        // 可选：显示检测结果（按ESC退出预览）
        cv::Mat display = frame.clone();
        cv::putText(display, "Valid: " + std::to_string(saved_count), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Calib Preview", display);
        if (cv::waitKey(50) == 27) break; // ESC键退出预览
    }

    cv::destroyAllWindows();
    std::cout << "\n🎉 处理完成！共筛选出 " << saved_count << " 张有效标定图像（输出目录：" << out_dir << "）" << std::endl;
    return saved_count;
}

