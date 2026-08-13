#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>      // 网络编程相关
#include <sys/socket.h>
#include <ros/ros.h> 
#include <iomanip>          // 输出精度
#include "test/gps.h"
#include "test/receivegpsresults.h" 
#include "test/yoloresults.h"

using namespace std;

#define BUF_LEN  100
char back[]="ok                 ";		//固定回复消息长度为20字节

const int PORT = 10000;          		//端口号#旧60888
const char *IP = "127.0.0.1"; 			//IP地址#旧"47.96.127.89"
struct sockaddr_in server_addr;         //定义服务端的地址结构体

// ========== 定点模式 ==========
bool use_fixed_target = false;          // 直接使用定点，默认关闭（false表示从socket接收）
double fixed_lat = 39.7252975;         // 工训楼前
double fixed_lon = 116.1678605;
// ==============================

void InitAddr()
{
	//std::cout<<"Initing"<<std::endl;
	bzero(&server_addr, sizeof(server_addr));//将地址结构体清零
	server_addr.sin_family = AF_INET;       //IPv4
	server_addr.sin_port = htons(PORT);		//将端口号转换为网络字节序
	//将服务器端IP转换成二进制地址填入结构体
	int ret = inet_pton(AF_INET, IP, &server_addr.sin_addr.s_addr);
	if (ret == -1)
	{
		perror("inet_pton error");
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "receive_gps");

	ros::NodeHandle n;
    ros::Publisher receivegps_pub = n.advertise<test::gps>("gps", 1);	//发给throw_ways投水坐标
	ros::Publisher result_pub = n.advertise<test::yoloresults>("yoloresults", 1);	//发给main_ctl，代表yolo检测完成收到目标
    ros::Publisher receivegps_result_pub = n.advertise<test::receivegpsresults>("receivegpsresults", 1);	//没用 发给main_ctl，代表receive_gps任务完成

	test::gps pub_gps;
	test::yoloresults yolo_result;
	test::receivegpsresults receivegps_results;	

    // ========== 定点模式 ==========
    if (use_fixed_target) {
        ROS_WARN("===== FIXED TARGET MODE =====");
        // 直接发布定点，然后进入循环保持节点存活
        yolo_result.flag = 1;
        result_pub.publish(yolo_result);

        receivegps_results.flag = 1;
        receivegps_result_pub.publish(receivegps_results);

        pub_gps.latitude = fixed_lat;
        pub_gps.longitude = fixed_lon;
        while (receivegps_pub.getNumSubscribers() == 0) {
            ROS_WARN("waiting for subscriber...");
            ros::spinOnce();  // 处理ROS事件，更新订阅者信息
            ros::Duration(0.1).sleep();
        }
        receivegps_pub.publish(pub_gps);
        ROS_INFO("Fixed GPS published!");

        // 保持节点运行（否则退出，影响主控）
        ros::Rate loop(1);
        while (ros::ok()) {
            ros::spinOnce();
            loop.sleep();
        }
        return 0;
    }
    // ==============================

    // 创建持久监听socket
    int listen_fd;
    if (-1 == (listen_fd = socket(AF_INET, SOCK_STREAM, 0)))// 创建TCP socket（监听）
    {
        ROS_ERROR("socket error: %s", strerror(errno));
        return -1;
    }
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    InitAddr();  // 填充server_addr
    if (bind(listen_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0)
    {
        ROS_ERROR("bind error: %s", strerror(errno));
        close(listen_fd);
        return -1;
    }
    if (listen(listen_fd, 5) < 0)
    {
        ROS_ERROR("listen error: %s", strerror(errno));
        close(listen_fd);
        return -1;
    }
    ROS_INFO("receive_gps listening on %s:%d", IP, PORT);

    while (ros::ok())
    {
        receivegps_results.flag = 0;
		receivegps_result_pub.publish(receivegps_results);

        // 接受一个客户端连接（阻塞）
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int ClientFd = accept(listen_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (ClientFd < 0)
        {
            ROS_ERROR("accept error: %s", strerror(errno));
            sleep(1);
            continue;
        }

        // 在同一个连接内接收三步数据
        double gps1[2] = {0};
        int step = 0;        // 0:等待纬度, 1:等待经度, 2:等待结束标志
        bool success = true;
        char Buf[BUF_LEN] = {0};

        while (step < 3 && ros::ok())
        {
            memset(Buf, 0, BUF_LEN);
            //接收客户端数据到buf
            int recv_len = recv(ClientFd, Buf, BUF_LEN, 0);
            if (recv_len <= 0)
            {
                printf("recv error or connection closed!\n");
                success = false;
                break;
            }

            double gps = atof(Buf);
            cout << setprecision(12) << "服务端发送的数据: " << gps << endl;

            //处理接收到的数据
            strcpy(back, "no                ");

            if (step == 0)  // 期待纬度
            {
                if (gps < 80.0 && gps > 10.0)   //范围判断是纬度
                {
                    gps1[0] = gps;
                    strcpy(back, "get lat success   ");
                    send(ClientFd, back, strlen(back), 0);  // 发送数据到客户端
                    step = 1;
                }
                else
                {
                    success = false;
                    break;
                }
            }
            else if (step == 1)  // 期待经度
            {
                if (gps > 90.0 && gps < 200.0 && gps1[0] != 0.0)
                {
                    gps1[1] = gps;
                    strcpy(back, "get lon success   ");
                    send(ClientFd, back, strlen(back), 0);
                    step = 2;
                }
                else
                {
                    success = false;
                    break;
                }
            }
            else if (step == 2)  // 期待结束标志
            {
                if (gps == 1.0 && gps1[1] != 0.0)  //收到结束标志且经纬度都已收到
                {
                    strcpy(back, "send success      ");
					send(ClientFd, back, strlen(back), 0);

                    //写入 + 发布
                    yolo_result.flag = 1;
                    result_pub.publish(yolo_result);

                    receivegps_results.flag = 1;
                    receivegps_result_pub.publish(receivegps_results);

                    pub_gps.latitude = gps1[0];
                    pub_gps.longitude = gps1[1];
                    ROS_WARN("Start waiting for subscriber");
                    
                    while (receivegps_pub.getNumSubscribers() == 0) //确保有订阅者再发gps，避免丢消息
                    {
                        ROS_WARN("waiting for subscriber... current: %d", receivegps_pub.getNumSubscribers());
                        ros::spinOnce();  // 处理ROS事件，更新订阅者信息
                        ros::Duration(0.1).sleep();
                    }
                    cout<<"about to publish GPS"<<endl;
                    //for(int i=0;i<10;i++)
                    //{
                        //receivegps_pub.publish(pub_gps);
                        //ros::spinOnce();
                        //ros::Duration(0.2).sleep();
                    //}
                    receivegps_pub.publish(pub_gps);
                    ROS_INFO("GPS published successfully");
                    
                    step = 3;   // 完成
                }
                else
                {
                    success = false;
                    break;
                }
            }
        }

        close(ClientFd);  // 关闭客户端套接字
        if (!success)
        {
            ROS_WARN("Failed to receive complete GPS data, resetting");
        }
        // 清空缓冲区
        memset(Buf, 0, BUF_LEN);
    }

    close(listen_fd);  // 关闭监听套接字
    return 0;
}