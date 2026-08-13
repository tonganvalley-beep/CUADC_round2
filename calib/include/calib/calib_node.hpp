#pragma once

#include "ChessCalib.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"


class CalibNode : public rclcpp::Node {

public:
    CalibNode(const rclcpp::NodeOptions &options);
    ~CalibNode() = default;

private:
    void imgCallback(const sensor_msgs::msg::Image::UniquePtr img_msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

    std::shared_ptr<ChessCalib> chess_calib;

};