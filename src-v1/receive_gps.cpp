#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <ros/ros.h> 
#include <iomanip>
#include "test/results.h" 
#define BUF_LEN  100
char back[]="ok                 ";
using namespace std;

const int PORT = 10000;          //端口号#旧60888
const char *IP = "127.0.0.1"; //IP地址#旧"47.96.127.89"
struct sockaddr_in server_addr;        //定义服务端的地址结构体


void InitAddr();

int main(int argc, char* argv[]) 
{
	ros::init(argc, argv, "rvgps_node");
	ros::NodeHandle nh;
	ros::Rate rate(20.0);  
    ros::Publisher pub_gps1=nh.advertise<test::results>("results",10);
	test::results pub_gps;

    int ServerFd, ClientFd;
    char Buf[BUF_LEN] = {0}; //缓冲区
    struct sockaddr ClientAddr;
    struct sockaddr_in numSockAddr;
    socklen_t addr_size = 0, recv_len = 0;
    int optval = 1;
    int count =0;

    double gps1[2]={0};
    while (ros::ok())
    {
		InitAddr();

		if (-1 == (ClientFd = socket(AF_INET, SOCK_STREAM, 0)))//创建套接字
		{
			ROS_ERROR("socket error: %s", strerror(errno)); // 输出具体错误信息
			sleep(1);
			continue;
		}
		/* 连接 */
		if (-1 == connect(ClientFd, (struct sockaddr*)&server_addr, sizeof(server_addr)))
		{
			printf("connect error!\n");
        	close(ClientFd);
			sleep(1);
			continue;
		}

        if ((recv_len = recv(ClientFd, Buf, BUF_LEN, 0)) < 0)//接收服务端数据的接口
        {
            printf("recv error!\n");
            exit(1);
        }
		double gps = atof(Buf);//atof ： ascll to double 字符串ascii码转换成double类型数据
      	cout<<setprecision(12)<<"服务端发送的数据: "<<gps<<endl;
      
      	strcpy(back,"no                ");
      	if(gps<80.0&&gps>10.0)//依据比赛范围的纬度特征判断发送过来的是经度还是纬度
		{ 
      		gps1[0]=gps; 
       		strcpy(back,"get lat success   ");
		}
        if(gps>90.0&&gps<200.0&&gps1[0]!=0.0)
		{ 
          	gps1[1]=gps;
          	strcpy(back,"get lon success   ");
        }
        if(gps==1.0&&gps1[1]!=0.0)//发送1.0是告诉客户端消息发完了 这里在判断gps发送的消息有没有结束
		{
          	pub_gps.latitude=gps1[0];
            pub_gps.longitude=gps1[1];
          	pub_gps.flag=1;
          	pub_gps1.publish(pub_gps);
           	strcpy(back,"send success      ");
          	gps1[0]=0;
           	gps1[1]=0;
		} 
      
        /* 发送数据到客户端 */
        send(ClientFd, back, strlen(back), 0);
        /* 关闭客户端套接字 */
        close(ClientFd);
        /* 清空缓冲区 */
        memset(Buf, 0, BUF_LEN);
    }
    return 0;
}

void InitAddr()
{
	//std::cout<<"Initing"<<std::endl;
	bzero(&server_addr, sizeof(server_addr));
	server_addr.sin_family = AF_INET;
	server_addr.sin_port = htons(PORT);
	//需要将服务器端的IP地址进行转换
	int ret = inet_pton(AF_INET, IP, &server_addr.sin_addr.s_addr);
	if (ret == -1)
	{
		perror("inet_pton error");
		exit(EXIT_FAILURE);
	}
}
