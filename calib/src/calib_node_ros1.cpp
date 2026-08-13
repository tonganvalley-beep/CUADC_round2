#include <ros/ros.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include "calib/ChessCalib.hpp"

void clearDirectory(const std::string& dir)
{
    DIR* dp = opendir(dir.c_str());

    // 如果目录不存在，则创建
    if (dp == nullptr)
    {
        mkdir(dir.c_str(), 0755);
        ROS_INFO("Created input folder: %s", dir.c_str());
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dp)) != nullptr)
    {
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0)
            continue;

        std::string file = dir + "/" + entry->d_name;
        remove(file.c_str());   // 删除普通文件
    }

    closedir(dp);
    ROS_INFO("Cleared input folder: %s", dir.c_str());
}

int main(int argc, char** argv) {
    // 初始化ROS1节点
    ros::init(argc, argv, "calib_node_ros1");
    ros::NodeHandle nh("~"); // 私有命名空间

    // 从ROS参数读取配置（替代命令行参数，更符合ROS习惯）
    int col, row;
    std::string img_dir, out_dir;
    nh.param<int>("col", col, 7);
    nh.param<int>("row", row, 7);
    nh.param<std::string>("img_dir", img_dir, "/home/tttian/calib_data/04"); // 输入文件夹
    nh.param<std::string>("out_dir", out_dir, "./valid_calib_images"); // 输出文件夹

	// 清空输出文件夹
    clearDirectory(out_dir);

    // 初始化标定类并处理文件夹
    ChessCalib calib(col, row, img_dir, out_dir);
    int valid_count = calib.processFolder();

    // 退出ROS节点
    ros::shutdown();
    return valid_count > 0 ? 0 : -1;
}

