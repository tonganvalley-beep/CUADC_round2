#include <ros/ros.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <string.h>
#include <stdint.h>

using namespace std;
using namespace cv;

static const int PORT = 9999;                       // 统一端口到 63101  端口实际上用的是9999
static const char* BIND_IP = "127.0.0.1";           // 监听所有网卡
static const char* IMG_DIR = "/home/tx2/Wintter/send_to_ground/";
static const int JPEG_QUALITY = 50;                 // 压缩质量（可调 1~100）
static const double LOOP_HZ = 10.0;                 // 发送循环频率

static sockaddr_in g_srv{};

// 安全写：直到全部写完
static bool send_all(int fd, const void* buf, size_t total) 
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);
    size_t sent = 0;
    while (sent < total) 
    {
        ssize_t n = ::send(fd, p + sent, total - sent, 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

static void InitAddr() 
{
    memset(&g_srv, 0, sizeof(g_srv));
    g_srv.sin_family = AF_INET;
    g_srv.sin_port = htons(PORT);
    if (inet_pton(AF_INET, BIND_IP, &g_srv.sin_addr.s_addr) != 1) 
    {
        perror("inet_pton error");
        exit(EXIT_FAILURE);
    }
}

static int create_listen_socket() 
{
    int fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd < 0) 
    {
        perror("socket");
        exit(1);
    }
    int opt = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) 
    {
        perror("setsockopt");
        exit(1);
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&g_srv), sizeof(g_srv)) < 0) 
    {
        perror("bind");
        exit(1);
    }
    if (listen(fd, 10) < 0) 
    {
        perror("listen");
        exit(1);
    }
    return fd;
}

static bool encode_jpeg(const cv::Mat& img, vector<unsigned char>& out, int quality = 50) 
{
    vector<int> params{IMWRITE_JPEG_QUALITY, quality};
    return cv::imencode(".jpg", img, out, params);
}

static void run() 
{
    ros::Rate rate(LOOP_HZ);
    int listen_fd = create_listen_socket();
    printf("pic_send_sky: listening on %s:%d\n", BIND_IP, PORT);

    while (ros::ok()) 
    {
        printf("\n000 等待客户端连接 (accept)…\n");
        sockaddr_in cli{};
        socklen_t alen = sizeof(cli);
        int cfd = accept(listen_fd, reinterpret_cast<sockaddr*>(&cli), &alen);
        if (cfd < 0) 
        {
            perror("accept");
            continue;
        }
        printf("111 客户端已连接\n");

        // 循环发送图片
        size_t idx = 0;
        while (ros::ok()) 
        {
            // 从 {idx}.jpg 读取
            std::string path = std::string(IMG_DIR) + std::to_string(idx) + ".jpg";
            cv::Mat frame = cv::imread(path, cv::IMREAD_COLOR);
            if (frame.empty()) 
            {
                // 没有新图就等下一轮
                rate.sleep();
                continue;
            }

            // 编码 JPEG
            vector<unsigned char> enc;
            if (!encode_jpeg(frame, enc, JPEG_QUALITY)) 
            {
                fprintf(stderr, "imencode failed: %s\n", path.c_str());
                idx++;
                rate.sleep();
                continue;
            }

            // 发送长度头（4 字节，网络序）
            uint32_t nlen = htonl(static_cast<uint32_t>(enc.size()));
            if (!send_all(cfd, &nlen, sizeof(nlen))) 
            {
                fprintf(stderr, "send length failed, drop client\n");
                break;
            }
            // 发送图像数据
            if (!send_all(cfd, enc.data(), enc.size())) 
            {
                fprintf(stderr, "send image failed, drop client\n");
                break;
            } 
            else 
            {
		        printf("send success!\n");
	        }

            idx++;
            rate.sleep();
        }
        ::close(cfd);
    }
    ::close(listen_fd);
}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "pic_send_node");
    ros::NodeHandle nh;
    InitAddr();
    run();
    return 0;
}
