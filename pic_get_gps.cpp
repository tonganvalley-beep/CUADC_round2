#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/NavSatFix.h>   	// GPS位置
#include <sensor_msgs/Imu.h>         	// IMU数据
#include <std_msgs/Float64.h>        	// 高度/航向角
#include <mavros_msgs/State.h>
#include <geometry_msgs/TwistStamped.h> //GPS、FCU速度类型名
#include <gst/gst.h>                 	// GStreamer
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <cmath>
#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

using namespace cv;
using namespace std;

//需要修改的变量
const int dis_lim = 45; 			//距离限制
double center_lon = 120.1103818;	//靶标区域中心的gps坐标
double center_lat = 30.5048068;
static double pipeline_delay_s = 0.00; // 可调的固定管线延迟（秒），用于将帧到达时间回溯到曝光中点
string exptr = "\"5000000 5000000\""; // 曝光时间范围（微秒） \"是一个转义符号

const int FPS = 5; 					//拍照帧率
const double dt = 1 / ((double)FPS);   	//更新时间步长
const int Width = 1920; 				//照片宽
const int Height = 1080; 				//照片高
const char* pic_path = "//home//tx2//Wintter//send_to_ground//"; 	//图片保存路径
const char* info_path = "//home//tx2//Wintter//info_to_ground//"; //信息保存路径
const char* raw_path = "//home//tx2//Wintter//raw_pic//"; 		//原图保存路径
int num_of_raw = 0;					 //记录储存照片数，用于保存图片时命名

//预设数值
double plane_lon = 116.1997506;	//飞机经度
double plane_lat = 39.5862045;	//飞机纬度
double lon_filter = 116.1631619;//最小二乘法后的飞机经度
double lat_filter = 39.7281999;	//最小二乘法后的飞机纬度
double lon_kf = 116.1631619;  	//数据处理后的飞机经度
double lat_kf = 39.7281999;		//卡尔曼滤波后的飞机纬度
double plane_alt = 15.79217;	//预设高度
double plane_yaw = 0.00;		//预设偏航角
double plane_pitch = 8.97877;	//预设俯仰角
double plane_roll = 8.576;		//预设滚转角
double last_roll = 0;			//用于计算滚转角差值
double plane_v_x = -24.46824689;   //预设东方向速度
double plane_v_y = 14.7066986; //预设北方向速度
double wx, wy, wz;	//滚转角速度，俯仰角速度，偏航角速度
float g1 = 0;		//用于四元数求解
float g2 = 0;
float g3 = 0;

//q11、q22为过程噪声的协方差矩阵对角线上的值，r11、r22为测量噪声的协方差矩阵对角线上的值
double p11, p12, p21, p22;
const double sigma2_a = 9;//加速度的方差
const double q11 = sigma2_a * pow(dt, 4) / 4, q12 = sigma2_a * pow(dt, 3) / 2, q21 = sigma2_a * pow(dt, 3) / 2, q22 = sigma2_a * pow(dt, 2);
const double r11 = 1, r22 = 0.09;
const double mavros_conf = 0.2;
const double t_delay_kf = -0.05;

double _lat = 0.0000091256386;	//0.000009009; 每向北走一米纬度增加：9.1256386×10e-6；
double _lon = 0.000011801996;		//0.00001136363; 每向东走一米经度增加：1.1801996×10e-5；
const double Earth_Radius = 6367967.0;
const double PI = 3.1415926535897932384626433832795;
const int array_len = 12;
double lon_array_kf[array_len] = { 0 };
double lat_array_kf[array_len] = { 0 };
double vx_kf[array_len] = { 0 };
double vy_kf[array_len] = { 0 };
double lon_filter_array[array_len] = { 0 };
double lat_filter_array[array_len] = { 0 };

struct TimedVal {
	double t;   // seconds
	double v;
};

static std::deque<TimedVal> buf_lat, buf_lon, buf_alt, buf_vx, buf_vy, buf_yaw, buf_pitch, buf_roll;
static std::mutex buf_mtx;
static const size_t BUF_MAX = 50; //最大储存量

static inline void push_buf(std::deque<TimedVal>& buf, double t, double v) 
{
	buf.emplace_back(TimedVal{ t, v });//在双端队列的末端就地创建（添加一个对象）这个对象之前可以不存在，push_back是先创建、拷贝再加入后删除之前创建的对象，在该性能更敏感的场景下使用emplace_back更为高效
	while (buf.size() > BUF_MAX) buf.pop_front();
}

// 线性插值;若不足则返回最近值
static double interp_at(const std::deque<TimedVal>& buf, double t) 
{
	if (buf.empty()) return 0.0;
	if (buf.size() == 1) return buf.back().v;
	// 如果 t 在范围外，返回最近端
	if (t <= buf.front().t) return buf.front().v;
	if (t >= buf.back().t) return buf.back().v;
	// 二分寻找
	size_t lo = 0, hi = buf.size() - 1;//索引 size_t就是一种索引，相当于unsigned int类型
	while (hi - lo > 1) 
	{
		size_t mid = (lo + hi) / 2;
		if (buf[mid].t < t) lo = mid; else hi = mid;
	}
	double t0 = buf[lo].t, t1 = buf[hi].t;
	double v0 = buf[lo].v, v1 = buf[hi].v;
	double a = (t - t0) / std::max(1e-6, (t1 - t0));
	return v0 + a * (v1 - v0);//线性插值
}

// GStreamer管道生成函数
std::string gstreamer_pipeline(int sensor_id = 0,
	int capture_width = 1920,
	int capture_height = 1080,
	int display_width = 1920,
	int display_height = 1080,
	int framerate = 20,
	int flip_method = 0) 
{
	return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) + //sensor_id指定物理摄像头编号 该参数一直到isdigitalgainrange为nvarguscamerasrc模块参数 光源采集与控制模块
		" exposuretimerange="+exptr+" gainrange=\"1 1\" ispdigitalgainrange=\"1 1\" exposurecompensation=0 aelock=true" + " ! "//exposuretimerange手动设置曝光时间范围 gainrange模拟增益范围锁定在最小增益减少噪点
		"video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" + std::to_string(capture_height) + ", "//ispdigitalgainrange数字增益范围 aelock = true自动曝光锁定 强制停止自动调亮 保证画面亮度恒定
		"format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) + "/1 ! "//video/x-raw(memory:NVMM)采集格式定义模块  width/height图像原始分辨率 format = NV12采用常见的YUV 4:2:0格式，适合硬件处理; framerate采集帧率 例如20/1为每秒20帧
		"nvvidconv flip-method=" + std::to_string(flip_method) + " ! "//nvvidconv 硬件转换/缩放模块 flip-method图像翻转/旋转 0:不转, 2:180°, 6:顺时针90°, 8:逆时针90° width/height输出分辨率 format = BGRx颜色空间转换 将YUV格式转换为带占位符的BGR格式
		"video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" + std::to_string(display_height) + ", format=(string)BGRx ! "//当然这里width和height实际上是video/x-raw模块里的参数，但是它会作用与前面这个模块进行强制转换
		"videoconvert ! "//videoconvert 通用转换与缓冲模块 
		"queue leaky=2 max-size-buffers=1 ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false";//！是连接各个模块的管道，可以看到相邻模块之间都有!连接
}//queue数据缓冲区模块 leaky丢弃策略 如果缓冲区满了，那就丢弃旧数据 max-size-buffers缓冲区大小，这里等于1那么意思就是只存一帧 appsink最终输出模块 drop = true 溢出处理，数据处理慢的时候直接丢弃新产生的帧
//sync = false禁用同步 不按时间戳等待让照片尽快地吐出来
void metersToLatLonWithAltitude(double &_lat, double &_lon) 
{
    // 纬度变化（每米对应的纬度差）
    _lat = 180.0 / (Earth_Radius * PI);

    // 经度变化（每米对应的经度差）
    _lon = 180.0 / (Earth_Radius * std::cos(center_lat * PI / 180.0) * PI);
}

sensor_msgs::NavSatFix global_pos;   //全球位置坐标回调函数
void gp_cb(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
	if (!msg)
	{
		ROS_ERROR("Null pointer in global_pos callback!");
		return;
	}
	global_pos = *msg;//备份数据到全局变量中，便于其它任何时候调用防止覆盖
	plane_lat = global_pos.latitude;
	plane_lon = global_pos.longitude;
	double tsec = msg->header.stamp.toSec();//尝试获取传感器产生数据的原始时间
	if (tsec == 0.0) tsec = ros::Time::now().toSec();//如果原始时间戳为零（该情况在模拟器或者某些硬件未同步的时候很常见），那么就使用当前系统的运行时间来代替
	std::lock_guard<std::mutex> lk(buf_mtx);//mutex 互斥锁 因为回调函数在独立的线程中运行，为了防止在主线程读取数据时发生"写冲突"，必须进行线程同步，但是怎么用其实不清楚
	push_buf(buf_lat, tsec, plane_lat);
	push_buf(buf_lon, tsec, plane_lon);
	return;
}

std_msgs::Float64 current_gps_hdg;    //GPS磁罗盘航向角回调函数
void gps_hdg_cb(const std_msgs::Float64::ConstPtr& msg)
{
	if (!msg)
	{
		ROS_ERROR("Null pointer in gps_hdg callback!");
		return;
	}
	current_gps_hdg = *msg;
	plane_yaw = current_gps_hdg.data;
	double tsec = ros::Time::now().toSec();
	std::lock_guard<std::mutex> lk(buf_mtx);
	push_buf(buf_yaw, tsec, plane_yaw);
	return;
}//和上面同样的逻辑

std_msgs::Float64 gps_rel_alt;   //GPS相对高度回调函数
void gps_rel_alt_cb(const std_msgs::Float64::ConstPtr& msg)
{
	if (!msg)
	{
		ROS_ERROR("Null pointer in gps_rel_alt callback!");
		return;
	}
	gps_rel_alt = *msg;
	plane_alt = gps_rel_alt.data;
	double tsec = ros::Time::now().toSec();
	std::lock_guard<std::mutex> lk(buf_mtx);
	push_buf(buf_alt, tsec, plane_alt);
	return;
}

sensor_msgs::Imu imu;   //IMU回调函数
void imu_cb(const sensor_msgs::Imu::ConstPtr& msg)
{
	if (!msg)
	{
		ROS_ERROR("Null pointer in IMU callback!");
		return;
	}
	imu = *msg;
	wx = imu.angular_velocity.x;//滚转角速度，向右为正
	wy = imu.angular_velocity.y;//俯仰角速度，低头为正
	wz = imu.angular_velocity.z;//偏航角速度，向西为正
	g1 = 2.0f * (imu.orientation.x * imu.orientation.z - imu.orientation.w * imu.orientation.y);
	g2 = 2.0f * (imu.orientation.w * imu.orientation.x + imu.orientation.y * imu.orientation.z);//这段公式我不是很记得，先搁置这个和视觉没什么关系
	g3 = imu.orientation.w * imu.orientation.w - imu.orientation.x * imu.orientation.x - imu.orientation.y * imu.orientation.y + imu.orientation.z * imu.orientation.z;
	plane_pitch = asinf(g1) * 57.3f;//asinf为asin的单精度浮点数形式，asin本身处理的是double形式，该形式运算速度更快，内存占用更小
	plane_roll = atanf(g2 / g3) * 57.3f;//
	double tsec = msg->header.stamp.toSec();
	if (tsec == 0.0) tsec = ros::Time::now().toSec();
	std::lock_guard<std::mutex> lk(buf_mtx);
	push_buf(buf_roll, tsec, plane_roll);
	push_buf(buf_pitch, tsec, plane_pitch);
	return;
}

geometry_msgs::TwistStamped gps_velocity;    //GPS速度参数回调函数
void gpsv_cb(const geometry_msgs::TwistStamped::ConstPtr& msg) 
{
	if (!msg) 
	{
        ROS_ERROR("Null pointer in GPS callback!");
        return;
    }
    gps_velocity = *msg;
    plane_v_x = gps_velocity.twist.linear.x;  // 获取东向速度
    plane_v_y = gps_velocity.twist.linear.y;  // 获取北向速度

    double tsec = ros::Time::now().toSec();
    if (tsec == 0.0) tsec = ros::Time::now().toSec(); // 确保时间戳有效

    std::lock_guard<std::mutex> lk(buf_mtx);
    // 将速度数据推送到缓冲区
    push_buf(buf_vx, tsec, plane_v_x);  // 将东向速度推送到 buf_vx
    push_buf(buf_vy, tsec, plane_v_y);  // 将北向速度推送到 buf_vy
}

// 修正后的DeleteFile函数
bool DeleteFile(const char* path)
{
	struct stat info;//stat 是一个对文件操作的结构体和函数 
	if (stat(path, &info) != 0)//通过文件路径，向系统查询文件消息，并将其存储到结构体中 本身返回0则获取成功 非零则获取失败
	{
		ROS_WARN("Path %s does not exist, skip deletion.", path);
		return true;
	}

	DIR* dir;//打开文件夹中的内容的指针 遍历文件夹中的成员
	struct dirent* dirinfo;//存储当前打开文件名的指针
	char filepath[512] = { 0 };  // 扩大缓冲区防止溢出

	if (S_ISREG(info.st_mode))//st.mode判断打开的是文件还是文件夹 判断是普通文件还是文件夹 像这里S_ISREG是在判断是否为文件 由此可推出为文件时返回值是非零
	{
		if (remove(path) == 0)//如果文件夹里面还有子文件夹，这是无法删除的，如果想要彻底删除还应该递归调用DeleteFile本身 当然这里已经是普通文件了所以不会有该情况
		{
			ROS_INFO("Removed file: %s", path);
			return true;
		}
		else
		{
			ROS_ERROR("Failed to remove file: %s", path);
			return false;
		}
	}
	else if (S_ISDIR(info.st_mode))
	{
		if ((dir = opendir(path)) == NULL)//opendir(path)依照路径打开文件夹并且返回一个DIR*,打开失败返回NULL
		{
			ROS_ERROR("Failed to open directory: %s", path);
			return false;
		}

		while ((dirinfo = readdir(dir)) != NULL)//readdir(dir)每调用一次，游标向下移动一步，并且返回下一个文件的信息到dirent*结构体中
		{
			if (strcmp(dirinfo->d_name, ".") == 0 || strcmp(dirinfo->d_name, "..") == 0)
				continue;

			// 安全拼接路径
			if (snprintf(filepath, sizeof(filepath), "%s/%s", path, dirinfo->d_name) >= sizeof(filepath))
			{
				ROS_ERROR("Path too long: %s/%s", path, dirinfo->d_name);
				continue;
			}

			if (remove(filepath) != 0)
			{
				ROS_ERROR("Failed to remove: %s", filepath);
			}
		}
		closedir(dir);//操作完成后关闭，否则将造成文件泄露等问题
		ROS_INFO("Cleaned directory: %s", path);
		return true;
	}
	return false;
}

double DistanceFromGPS(double lat_a, double lon_a, double lat_b, double lon_b)
{
	double n = (double)(180 / M_PI);

	double a1 = lat_a / n;
	double a2 = lon_a / n;
	double b1 = lat_b / n;
	double b2 = lon_b / n;

	double t1 = cos(a1) * cos(a2) * cos(b1) * cos(b2);
	double t2 = cos(a1) * sin(a2) * cos(b1) * sin(b2);
	double t3 = sin(a1) * sin(b1);
	double t = acos(t1 + t2 + t3);

	return 6366000 * t;
}//这个是什么公式吧，但是具体不清楚

double Kalman_Filter(double* a1, double* a2, double _u)
{
	double x0 = a1[0];
	double z1[array_len] = { 0 }, z2[array_len] = { 0 };//z1为广义坐标测量值，z2为广义速度测量值
	for (int j = 0; j < array_len; j++)
	{
		z1[j] = (a1[j] - a1[0]) / _u;
	}

	for (int m = 0; m < array_len; m++)
	{
		z2[m] = a2[m];
	}

	//后验估计协方差矩阵初始化
	p11 = 16;
	p12 = p21 = 0;
	p22 = 0.16;

	double x1[array_len] = { 0 }, x2[array_len] = { 0 };//x1、x2分别为存储广义坐标、广义速度后验估计值的数组
	x1[0] = z1[0];
	x2[0] = z2[0];

	double _x1[array_len] = { 0 }, _x2[array_len] = { 0 };//_x1、_x2分别为存储广义坐标、广义速度先验估计值的数组
	_x1[0] = x1[0];
	_x2[0] = x2[0];

	for (int k = 1; k < array_len; k++)
	{
		/****************预测*****************/
		//首先计算先验估计值
		_x1[k] = x1[k - 1] + x2[k - 1] * dt;
		_x2[k] = x2[k - 1];

		//计算先验估计的协方差矩阵
		double _p11, _p12, _p21, _p22;
		_p11 = p11 + (p12 + p21) * dt + p22 * dt * dt + q11;
		_p12 = _p21 = p12 + p22 * dt + q12;
		_p22 = p22 + q22;

		/****************根据测量值（将从mavros获取数据当作测量值）来进行校正********************/
		//首先计算Kalman因数Kk
		double s11, s12, s21, s22;//S为新息协方差矩阵的值
		s11 = _p11 + r11;
		s12 = s21 = _p12;
		s22 = _p22 + r22;

		double det_S = s11 * s22 - s12 * s21;
		double Kk11, Kk12, Kk21, Kk22;//Kk为Kalman矩阵的值
		Kk11 = (_p11 * s22 - _p12 * s21) / det_S;
		Kk12 = (-_p11 * s12 + _p12 * s11) / det_S;
		Kk21 = (_p21 * s22 - _p22 * s21) / det_S;
		Kk22 = (-_p21 * s12 + _p22 * s11) / det_S;

		//校正
		x1[k] = _x1[k] + Kk11 * (z1[k] - _x1[k]) + Kk12 * (z2[k] - _x2[k]);
		x2[k] = _x2[k] + Kk21 * (z1[k] - _x1[k]) + Kk22 * (z2[k] - _x2[k]);

		//更新后验估计的协方差矩阵
		p11 = (1 - Kk11) * _p11 - Kk12 * _p21;
		p21 = p12 = (1 - Kk11) * _p12 - Kk12 * _p22;
		p22 = -Kk21 * _p12 + (1 - Kk22) * _p22;
	}

	double pos_kf = x0 + x1[array_len - 1] * _u + t_delay_kf * x2[array_len - 1] * _u;
	return pos_kf;
}

//尝试用最小二乘法来修正经纬度坐标
double get_pos(double* a, double cur)
{
	for (int i = 0; i < array_len - 1; i++)
	{
		a[i] = a[i + 1];
	}
	a[array_len - 1] = cur;

	double pos[array_len] = { 0 };
	for (int i = 1; i < array_len; i++)
	{
		pos[i] = a[i] - a[0];
	}

	double pos_adv = 0, x_adv = (array_len - 1) / 2;
	for (int i = 0; i < array_len; i++)
	{
		pos_adv = pos_adv + pos[i];
	}
	pos_adv = pos_adv / array_len;

	double k, b, sum1 = 0, sum2 = 0;
	for (int n = 0; n < array_len; n++)
	{
		sum1 = sum1 + n * pos[n];
	}
	sum1 = sum1 - array_len * x_adv * pos_adv;

	for (int m = 0; m < array_len; m++)
	{
		sum2 = sum2 + m * m;
	}
	sum2 = sum2 - array_len * x_adv * x_adv;

	k = sum1 / sum2;
	b = pos_adv - k * x_adv;

	double pos_filter = a[0] + k * (array_len - 1) + b;
	return pos_filter;
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
}//没看懂

int main(int argc, char** argv)
{
	ros::init(argc, argv, "pic_get_node");
	ros::NodeHandle nh;

	// 删除路径下的所有文件
	if (!DeleteFile(pic_path) || !DeleteFile(info_path) || !DeleteFile(raw_path)) 
	{
		ROS_FATAL("Failed to initialize directories!");
		return -1;
	}

	metersToLatLonWithAltitude(_lat,_lon);

	// 初始化全局数组为初始GPS值（而非全0）
	for (int i = 0; i < array_len; i++) 
	{
		lat_array_kf[i] = global_pos.latitude;//已经创建好的全局变量
		lon_array_kf[i] = global_pos.longitude;
		vy_kf[i] = plane_v_y;
		vx_kf[i] = plane_v_x;
	}

	// 等待关键话题数据就绪
	ROS_INFO("Waiting for MAVROS topics...");
	// 等待MAVROS数据（使用显式类型声明避免auto冲突）
	boost::shared_ptr<const sensor_msgs::NavSatFix> global_pos_msg =
		ros::topic::waitForMessage<sensor_msgs::NavSatFix>("mavros/global_position/global", nh);
	if (!global_pos_msg) 
	{
		ROS_FATAL("MAVROS global position topic not available!");
		return -1;
	}

	// 订阅话题
	ros::Subscriber gp_sub = nh.subscribe<sensor_msgs::NavSatFix>     //订阅全球GPS坐标
		("mavros/global_position/global", 1, gp_cb);
	ros::Subscriber gps_rel_alt_sub = nh.subscribe<std_msgs::Float64>     //订阅GPS相对高度
		("mavros/global_position/rel_alt", 1, gps_rel_alt_cb);
	ros::Subscriber gps_hdg_sub = nh.subscribe<std_msgs::Float64>      //订阅GPS磁罗盘航向角
		("mavros/global_position/compass_hdg", 1, gps_hdg_cb);
    ros::Subscriber gpsv_sub = nh.subscribe<geometry_msgs::TwistStamped>      //订阅GPS速度参数
        ("mavros/global_position/raw/gps_vel", 1, gpsv_cb);
	ros::Subscriber imu_sub = nh.subscribe<sensor_msgs::Imu>         //订阅IMU数据
		("mavros/imu/data", 1, imu_cb);

	//等待必要话题连接
	ros::Rate rate(20.0);
	for (int i = 50;ros::ok() && i > 0;i--) 
	{
		ros::spinOnce();
		rate.sleep();
		ROS_INFO("AOA is initializing , waiting for all connections to establish");
	}

	// 修改后的GStreamer初始化
	if (!gst_is_initialized())//这个找不到定义？这个在哪里定义的
	{
		gst_init(nullptr, nullptr);
		if (!gst_is_initialized())
		{
			ROS_FATAL("Failed to initialize GStreamer!");
			return -1;
		}
	}

	// 创建GStreamer管道
	string pipeline = gstreamer_pipeline(0, Width, Height, Width, Height, FPS);
	cout << "Using pipeline: " << pipeline << endl;

	// 打开摄像头
	VideoCapture capture(pipeline, CAP_GSTREAMER);

	if (!capture.isOpened())
	{
		ROS_ERROR("Camera open failed!");
		return -1;
	}

	printf("拍照距离限制=%d\n", dis_lim);

	ros::Rate rate_pic(FPS); //拍摄照片的频率
	Mat current_frame;

	while (ros::ok())
	{
		ros::spinOnce();
		capture >> current_frame; //从相机获取一帧图像，紧跟spinOnce保证数据实时

		if (current_frame.empty())
		{
			ROS_ERROR("Failed to capture image from camera!");
			continue;
		}

		// 帧时间 & 曝光时间（回溯固定管线延迟）
		double t_frame = ros::Time::now().toSec();
		double t_expo = t_frame - pipeline_delay_s;

		// 取曝光时刻的插值位姿/速度
		{
			std::lock_guard<std::mutex> lk(buf_mtx);
			plane_lat = interp_at(buf_lat, t_expo);
			plane_lon = interp_at(buf_lon, t_expo);
			plane_alt = interp_at(buf_alt, t_expo);
			plane_yaw = interp_at(buf_yaw, t_expo);
			plane_pitch = interp_at(buf_pitch, t_expo);
			plane_roll = interp_at(buf_roll, t_expo);

			// 插值处理飞机的速度
			plane_v_x = interp_at(buf_vx, t_expo);  // 插值东向速度
			plane_v_y = interp_at(buf_vy, t_expo);  // 插值北向速度
		}

		//更新数据组数据，用于滤波之类的修正
		for (int i = 0; i < array_len - 1; i++)
		{
			lat_array_kf[i] = lat_array_kf[i + 1];
			lon_array_kf[i] = lon_array_kf[i + 1];
			vy_kf[i] = vy_kf[i + 1];
			vx_kf[i] = vx_kf[i + 1];
		}
		lat_array_kf[array_len - 1] = plane_lat;
		lon_array_kf[array_len - 1] = plane_lon;
		vy_kf[array_len - 1] = plane_v_y;
		vx_kf[array_len - 1] = plane_v_x;

		// 最小二乘法修正经纬度坐标
		lat_filter = get_pos(lat_filter_array, plane_lat);
		lon_filter = get_pos(lon_filter_array, plane_lon);

		// 卡尔曼滤波修正经纬度坐标
		lat_kf = Kalman_Filter(lat_array_kf, vy_kf, _lat);
		lon_kf = Kalman_Filter(lon_array_kf, vx_kf, _lon);

		double dis = DistanceFromGPS(center_lat, center_lon, plane_lat, plane_lon);
		printf("......%lf......\n", dis);

		if (dis < dis_lim && plane_alt > 10)
		{
			// 批量写入信息
			std::vector<std::pair<double, int>> info_list = {
				{lon_filter, 0}, 
				{lat_filter, 10}, 
				{plane_alt, 20},
				{plane_pitch, 30}, 
				{plane_yaw, 40}, 
				{plane_roll, 50},
				{lon_kf, 60}, 
				{lat_kf, 70},
				{plane_lon, 80}, 
				{plane_lat, 90}, 
				{plane_v_x, 100}, 
				{plane_v_y, 110}
			};
			Mat cell_raw = Mat::zeros(Size(150, 8), CV_8UC3);//zeros imwrite capture这些的作用是什么
			cell_raw = Scalar(48, 48, 48);
			write_all_info(cell_raw, info_list);

			//保存天井
			imwrite(raw_path + to_string(num_of_raw) + ".jpg", current_frame); 
			// 保存信息图片
			imwrite(info_path + to_string(num_of_raw) + ".png", cell_raw);

			num_of_raw++;
		}

		rate_pic.sleep();
	}

	capture.release();
	return 0;
}
