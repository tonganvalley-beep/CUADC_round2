#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/WaypointReached.h>
#include <tf/tf.h>
#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <deque>
#include <thread>
#include <mutex>
#include <chrono>
#include <signal.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include "test/savepicresults.h"

using namespace cv;
using namespace std;

//需要修改的变量
const int dis_lim = 50; 			    //距离限制
string exptr = "\"5000000 5000000\"";   //曝光时间范围（微秒）
int circle_wp = 12;                      //盘旋航点

double center_lon = 116.1997211;	    //靶标区域中心的gps坐标
double center_lat = 39.5863099;
bool ground_test = true;				//是否处于地面测试状态
const int FPS = 5;              		//拍照帧率
const int WIDTH = 1920;         		//照片宽
const int HEIGHT = 1080;        		//照片高
#define SHM_NAME "/my_shm"      		//共享内存
volatile sig_atomic_t g_running = 1;	//清理代码

const char* info_path = "//home//tx2//Wintter//info_to_ground//";   //信息保存路径
const char* raw_path = "//home//tx2//Wintter//raw_pic//"; 	        //原图保存路径
const char* pic_path = "//home//tx2//Wintter//send_to_ground//";    //图片保存路径

//飞机状态
double plane_lat = 0, plane_lon = 0, plane_alt = 0;     //纬经高
double plane_yaw = 0, plane_pitch = 0, plane_roll = 0;  //偏航、俯仰、滚转
int current_wp = -1;                                    //当前航点编号

//飞机高度相关
double alt_takeoff = 0.0;
bool takeoff_set = false;
double alt_sum = 0.0;
int alt_count = 0;

//共享内存
#pragma pack(push,1)
struct SharedData
{
    int flag;   //0代表空闲（可以写），1代表正在写，2代表已写完（可以读）
    int frame_id;
    int wp;
    double lon, lat, alt;
    double pitch,yaw,roll;
    unsigned char image[WIDTH*HEIGHT*3];
};
#pragma pack(pop)

SharedData* shm_ptr;

//清空目录
bool ClearDirectory(const char* path)
{
    DIR* dir = opendir(path);
    if (!dir) return false;

    struct dirent* entry;
    char filepath[512];

    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_name[0] == '.') continue;
        snprintf(filepath,sizeof(filepath),"%s/%s",path,entry->d_name);
        remove(filepath);
    }
    closedir(dir);
    return true;
}

//距离计算
double DistanceFromGPS(double lat_a,double lon_a,double lat_b,double lon_b)
{
    double n = 180.0/M_PI;
    double a1=lat_a/n,a2=lon_a/n,b1=lat_b/n,b2=lon_b/n;

    double val = cos(a1)*cos(a2)*cos(b1)*cos(b2)
               + cos(a1)*sin(a2)*cos(b1)*sin(b2)
               + sin(a1)*sin(b1);

    val = min(1.0,max(-1.0,val));
    return 6366000 * acos(val);
}

//GStreamer管道生成函数
std::string gstreamer_pipeline(int sensor_id = 0,
	int capture_width = 1920,
	int capture_height = 1080,
	int display_width = 1920,
	int display_height = 1080,
	int framerate = 20,
	int flip_method = 0) 
{
	return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) +
		" exposuretimerange="+exptr+" gainrange=\"1 1\" ispdigitalgainrange=\"1 1\" exposurecompensation=0 aelock=true" + " ! "
		"video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" + std::to_string(capture_height) + ", "
		"format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) + "/1 ! "
		"nvvidconv flip-method=" + std::to_string(flip_method) + " ! "
		"video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" + std::to_string(display_height) + ", format=(string)BGRx ! "
		"videoconvert ! "
		"queue leaky=2 max-size-buffers=1 ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false";
}

//写入信息
void write_all_info(Mat& frame, const vector<pair<double, int>>& info_list) 
{
	int ch = frame.channels();
	int width = frame.cols;
	uchar* data = frame.ptr<uchar>(0);

	for (const auto& info : info_list) 
	{
		int b = info.second;
		if (b < 0 || b + 10 > width) 
		{
			ROS_ERROR("write_info: Position %d is out of bounds (width=%d)", b, width);
			continue;
		}

		string s = to_string(info.first);
		for (int i = b; i < b + 10; i++) 
		{
			if ((i - b) > s.length() - 1) break;
			data[i * ch] = static_cast<uchar>(s[i - b]);
		}
	}
}

//MAVROS信息获取
void gps_cb(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
    plane_lat = msg->latitude;
    plane_lon = msg->longitude;
}

void pos_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    tf::Quaternion q(
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z,
        msg->pose.orientation.w);

    tf::Matrix3x3 m(q);
    m.getRPY(plane_roll,plane_pitch,plane_yaw);

    double alt_now = msg->pose.position.z;

    //初始高度
    if (!takeoff_set)
    {
        alt_sum += alt_now;
        alt_count++;

        //累积帧，取均值
        if (alt_count >= 50)
        {
            alt_takeoff = alt_sum / alt_count;
            takeoff_set = true;
        }

        return;
    }

    //相对高度
    plane_alt = alt_now - alt_takeoff;
}

void wp_cb(const mavros_msgs::WaypointReached::ConstPtr& msg)
{
    current_wp = msg->wp_seq;
}

void shutdown_handler(int sig)
{
	g_running = 0;
}

int main(int argc,char** argv)
{
    ros::init(argc,argv,"pic_get_node");
    ros::NodeHandle nh;

	// --- 加载配置文件 ---
    std::string config_path = "/home/tx2/Wintter/src/test/config.yaml";
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
        ROS_INFO("配置文件加载成功: %s", config_path.c_str());
    } catch (const YAML::Exception& e) {
        ROS_FATAL("加载配置文件失败: %s  错误信息: %s", config_path.c_str(), e.what());
        return -1;
    }
    ground_test = config["ground_test"].as<bool>();
    center_lat = config["throw_target"]["lat"].as<double>();
    center_lon = config["throw_target"]["lon"].as<double>();

    //删除路径下的所有文件
	if (!ClearDirectory(pic_path) || !ClearDirectory(info_path) || !ClearDirectory(raw_path)) 
	{
		ROS_FATAL("Failed to initialize directories!");
		return -1;
	}

	if(!ground_test)
	{
		//等待关键话题数据就绪
		ROS_INFO("Waiting for MAVROS topics...");
		//等待MAVROS数据（使用显式类型声明避免auto冲突）
		boost::shared_ptr<const sensor_msgs::NavSatFix> global_pos_msg =
			ros::topic::waitForMessage<sensor_msgs::NavSatFix>("mavros/global_position/global", nh);
		if (!global_pos_msg) 
		{
			ROS_FATAL("MAVROS global position topic not available!");
			return -1;
		}
	}

    ros::Subscriber gps_sub = nh.subscribe("mavros/global_position/global",1,gps_cb);     //订阅全球GPS坐标
    ros::Subscriber pos_sub = nh.subscribe("/mavros/local_position/pose",1,pos_cb);       //订阅飞机姿态角数据
    ros::Subscriber wp_sub  = nh.subscribe("mavros/mission/reached",1,wp_cb);             //订阅航点
	
	ros::Publisher savepicresults_pub = nh.advertise<test::savepicresults>("savepicresults", 10);
	test::savepicresults savepic_results;

	signal(SIGUSR1,shutdown_handler);

	//等待必要话题连接
	ros::Rate rate(20.0);
	for (int i = 50;ros::ok() && i > 0;i--) 
	{
		ros::spinOnce();
		rate.sleep();
		ROS_INFO("AOA is initializing , waiting for all connections to establish");
	}

    //共享内存
    int fd = shm_open(SHM_NAME,O_CREAT|O_RDWR,0666);
    ftruncate(fd,sizeof(SharedData));
    shm_ptr = (SharedData*)mmap(NULL,sizeof(SharedData),
        PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    shm_ptr->flag = 0;

    //GStreamer初始化
	if (!gst_is_initialized())
	{
		gst_init(nullptr, nullptr);
		if (!gst_is_initialized())
		{
			ROS_FATAL("Failed to initialize GStreamer!");
			return -1;
		}
	}

	//创建GStreamer管道
	string pipeline = gstreamer_pipeline(0, WIDTH, HEIGHT, WIDTH, HEIGHT, FPS);
	cout << "Using pipeline: " << pipeline << endl;

    //打开摄像头
	VideoCapture cap(pipeline,CAP_GSTREAMER);

    if(!cap.isOpened())
    {
        ROS_ERROR("camera failed");
        return -1;
    }

    printf("拍照距离限制=%d\n", dis_lim);

    int id=0;
    ros::Rate pic_rate(5);

    while(ros::ok() && g_running)
    {
        Mat frame;
        if(!cap.read(frame))
        {
			if(!g_running) break;
            ROS_ERROR("Failed to capture image from camera!");
            continue;
        }

        ros::spinOnce();

        double dis = DistanceFromGPS(center_lat,center_lon,plane_lat,plane_lon);
        printf("......%lf......\n", dis);

        if((dis<dis_lim && plane_alt>10) || current_wp>=circle_wp || ground_test)
        {
            id++;

            //共享内存
            if(shm_ptr->flag==0)
            {
                shm_ptr->flag=1;

                shm_ptr->frame_id=id;
                shm_ptr->wp=current_wp;
                shm_ptr->lon=plane_lon;
                shm_ptr->lat=plane_lat;
                shm_ptr->alt=plane_alt;
                shm_ptr->pitch=plane_pitch;
                shm_ptr->yaw=plane_yaw;
                shm_ptr->roll=plane_roll;

                memcpy(shm_ptr->image,frame.data,WIDTH*HEIGHT*3);

                shm_ptr->flag=2;
            }
        }

        pic_rate.sleep();
    }
	printf("Publishing flags...\n");
	savepic_results.flag = 1;
	savepicresults_pub.publish(savepic_results);
	ros::spinOnce();
	ros::Duration(0.5).sleep();

    //释放摄像头
	if(cap.isOpened())
	{
		cap.release();
		printf("camera released by signal\n");
	}

    //释放共享内存映射
    if (shm_ptr != NULL && shm_ptr != MAP_FAILED) 
	{
        munmap(shm_ptr, sizeof(SharedData));
        printf("Shared memory unmapped\n");
    }
    
    //关闭共享内存文件描述符
    if (fd > 0) 
	{
        close(fd);
        printf("Shared memory fd closed\n");
    }
    
    //删除共享内存
    shm_unlink(SHM_NAME);
    printf("Shared memory unlinked\n");
    
    //清理 GStreamer
    if (gst_is_initialized()) 
	{
        gst_deinit();
        printf("GStreamer deinitialized\n");
    }
    
    ros::shutdown();
    printf("Cleanup complete, exiting...\n");

	return 0;
}
