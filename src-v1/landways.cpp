#include <iostream>
#include <cmath>
#include <ros/ros.h> 
#include <unistd.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/Float64.h>
#include <mavros_msgs/WaypointPush.h>
#include <mavros_msgs/RCIn.h>  
#include <mavros_msgs/RCOut.h> 
#include "test/results.h"  
using namespace std;
int Whether_use_current = 0;
int Whether_appoint_choice = 1;
int Land_choice = 2;//0:rr 1:ll 2:rl 3:lr

double latitude = 30.5048068;
double longitude = 120.1103818;
double current_yaw = 80.0;//正北为0，顺时针为正，单位为角度制
const double R1 = 25;
const double length1 = 50;
const int num_waypoints_1 = 5;

double land_lat = 30.5044497;
double land_lon = 120.1083876;
double land_yaw = 265.0;//正北为0，顺时针为正，单位为角度制
const double R2 = 25;
const double length2 = 100;
const int num_waypoints_2 = 3;

double current_yaw_rad = 0.0;
double land_yaw_rad = 0.0;
double _lat = 0.0000091256386;
double _lon = 0.00001180199;
const double PI = 3.1415926535897932384626433832795;
const double Earth_Radius = 6367967.0;
const double latitude_now = (latitude + land_lat) / 2;//为什么要除以2？
const int do_jump_waypoints_num = 45;
int flag = 0;

void metersToLatLonWithAltitude(double &_lat, double &_lon) {
    // 纬度变化（每米对应的纬度差）
    _lat = 180.0 / (Earth_Radius * PI);

    // 经度变化（每米对应的经度差）
    _lon = 180.0 / (Earth_Radius * std::cos(latitude_now * PI / 180.0) * PI);
}

mavros_msgs::RCIn rcin;   //遥控器输入回调函数
void rcin_cb(const mavros_msgs::RCIn::ConstPtr& msg) {
	rcin = *msg;
	if(rcin.channels[10]>1800)//11通pwm表明判断是否已经投水
		flag = 1;
}

mavros_msgs::RCOut rcout;   //舵机输出回调函数
void rcout_cb(const mavros_msgs::RCOut::ConstPtr& msg) {
	rcout = *msg;
	if(rcout.channels[6]>1800)//7通pwm表明判断是否已经投水


		flag = 1;
}

int result_flag = 0;
test::results result;
void receive(const test::results::ConstPtr& msg)
{
	result = *msg;
	if(result_flag == 0) {
		if(Whether_use_current == 0) {
			latitude = result.latitude;
			longitude = result.longitude;
		}
		result_flag = 1;
	}
}

int gps_flag = 0;
sensor_msgs::NavSatFix global_pos;
void gp_cb(const sensor_msgs::NavSatFix::ConstPtr& msg) {
	if (!msg) {
        ROS_ERROR("Null pointer in GPS callback!");
        return;
    }//以防万一指针信息丢失变成空指针
	global_pos = *msg;
	if(flag == 1 && gps_flag == 0) {
		if(Whether_use_current == 1) {
			latitude = global_pos.latitude;
			longitude = global_pos.longitude;
		}
		gps_flag = 1;
	}
}

int yaw_flag = 0;
std_msgs::Float64 current_gps_hdg;    //GPS磁罗盘航向角回调函数
void gps_hdg_cb(const std_msgs::Float64::ConstPtr& msg) {
	if (!msg) {
        ROS_ERROR("Null pointer in GPS callback!");
        return;
    }
    current_gps_hdg = *msg;
	if(flag == 1 && yaw_flag == 0) {
		if(Whether_use_current == 1) {
			current_yaw = current_gps_hdg.data;
		}
		current_yaw_rad = current_yaw * PI / 180.0;
		yaw_flag = 1;
	}
}

int main(int argc, char* argv[]) {
	ros::init(argc, argv, "land_node");
	ros::NodeHandle n;
	ros::Rate rate(20.0);  //更新速度20HZ

	mavros_msgs::WaypointPush::Request request;
	mavros_msgs::WaypointPush::Response response;//mavros_msg的服务消息等等复习一下
	mavros_msgs::Waypoint w,w1,w2,w3;

	ros::ServiceClient land_Client = n.serviceClient<mavros_msgs::WaypointPush>
		("mavros/mission/push");
	ros::Subscriber rcin_sub = n.subscribe<mavros_msgs::RCIn>     //订阅遥控器输入数据
		("mavros/rc/in", 1, rcin_cb);
	ros::Subscriber rcout_sub = n.subscribe<mavros_msgs::RCOut>     //订阅舵机输出数据
		("mavros/rc/out", 1, rcout_cb);
	ros::Subscriber gp_sub = n.subscribe<sensor_msgs::NavSatFix>     //订阅全球GPS坐标
		("mavros/global_position/global", 1, gp_cb);//ROS订阅里面塞函数的意思是啥来着，消息怎么传递的 publisher发布ROS自动调用回调函数把消息存进去
	ros::Subscriber gps_hdg_sub = n.subscribe<std_msgs::Float64>      //订阅GPS磁罗盘航向角
        ("mavros/global_position/compass_hdg", 1, gps_hdg_cb);
	ros::Subscriber person_info_sub = n.subscribe<test::results>("results", 10, receive);

	for (int i = 100; ros::ok() && i > 0; --i) 
	{
		ros::spinOnce();//主动跑一次ROS自己的所有消息队列和回调函数
		rate.sleep();//rate.sleep()按照频率等一会儿 
	}//预先使用老办法消耗100/20=5s的时间，使得ros的callback能够真正启动并创建数据

	cout << "Waiting for land." << endl;

	int waiting = 0;
	int prepared_flag = 0;
	while (ros::ok())
	{
		if(prepared_flag == 0) {
			metersToLatLonWithAltitude(_lat, _lon);
			if(current_yaw>=0 && current_yaw<=180) current_yaw_rad = current_yaw * PI / 180.0;//正北为零，顺时针为正，角度制
			else current_yaw_rad = (current_yaw-360) * PI / 180.0;//是按照正弧度负弧度这样理解吗 是的实际上就是在考虑方向
			if(land_yaw>=0 && land_yaw<=180) land_yaw_rad = land_yaw * PI / 180.0;
			else land_yaw_rad = (land_yaw-360) * PI / 180.0;
			double land_preparing_lat_1 = land_lat - cos(land_yaw_rad)*length2*_lat;//降落方向往回100m得到准备降落点
			double land_preparing_lon_1 = land_lon - sin(land_yaw_rad)*length2*_lon; 
			if(Whether_appoint_choice == 1 && Land_choice == 2) {//左右 右左 右右 左左的降落方式有什么区别？
				double center_lat_start_right_1 = latitude + (length1 - 30) * _lat * cos(current_yaw_rad) - R1 * _lat * sin(current_yaw_rad);//从latitude(这个对应的具体位置是什么)往当前机头朝向飞70m，然后从这个地方开始相切求第一个圆的圆心坐标
				double center_lat_end_left_1 = land_preparing_lat_1 + R2 * _lat * sin(land_yaw_rad);//准备降落点的位置开始相切，求第二个圆的圆心
				double center_lon_start_right_1 = longitude + (length1 - 30) * _lon * sin(current_yaw_rad) + R1 * _lon * cos(current_yaw_rad);
				double center_lon_end_left_1 = land_preparing_lon_1 - R2 * _lon * cos(land_yaw_rad);
				double center_distance_rl_1 = sqrt(pow((center_lat_start_right_1-center_lat_end_left_1)/_lat, 2) + pow((center_lon_start_right_1-center_lon_end_left_1)/_lon, 2));//求两个圆的圆心距离
				if(center_distance_rl_1 > R1 + R2) {
					prepared_flag = 1;//如果两个圆不相切那么就便于找到两个圆之间的公切线
				}
				else {
					ROS_ERROR("Landways could not prepared.");
				}
			}
			else if(Whether_appoint_choice == 1 && Land_choice == 3) {//
				double center_lat_end_right_1 = land_preparing_lat_1 - R2 * _lat * sin(land_yaw_rad);
				double center_lon_end_right_1 = land_preparing_lon_1 + R2 * _lon * cos(land_yaw_rad); 
				double center_lat_start_left_1 = latitude + (length1 - 30) * _lat * cos(current_yaw_rad) + R1 * _lat * sin(current_yaw_rad);
				double center_lon_start_left_1 = longitude + (length1 - 30) * _lon * sin(current_yaw_rad) - R1 * _lon * cos(current_yaw_rad);
				double center_distance_lr_1 = sqrt(pow((center_lat_start_left_1-center_lat_end_right_1)/_lat, 2) + pow((center_lon_start_left_1-center_lon_end_right_1)/_lon, 2));
				if(center_distance_lr_1 > R1 + R2) {
					prepared_flag = 1;
				}
				else {
					ROS_ERROR("Landways could not prepared.");
				}
			}//预判断 生成的圆弧航线是否安全（也就是说两段圆弧航线是否相离）然后用公切线连接两段圆弧
		}
		if(((gps_flag == 1 && yaw_flag == 1) || (result_flag == 1)) && waiting == 0)
		{
			double land_preparing_lat = land_lat - cos(land_yaw_rad)*length2*_lat;
			double land_preparing_lon = land_lon - sin(land_yaw_rad)*length2*_lon; 
			double theta_need = 0.5 * PI - atan2((land_preparing_lat - latitude + length1 * _lat * cos(current_yaw_rad))/_lat, (land_preparing_lon - longitude + length1 * _lon * sin(current_yaw_rad))/_lon);
			int j = 0, choice = 0, count_1 = 0, count_2 = 0;
			double latP1[100] = { 0 }, lonP1[100] = { 0 }, latP2[100] = { 0 }, lonP2[100] = { 0 };
			double center_lat_start_right, center_lon_start_right, center_lat_end_right, center_lon_end_right;
			double center_lat_start_left, center_lon_start_left, center_lat_end_left, center_lon_end_left;
			double center_theta_rr, center_theta_rl, center_theta_lr, center_theta_ll;
			double center_distance_rl, center_distance_lr; 
			double theta_circle_rr_1, theta_circle_rr_2, theta_circle_ll_1, theta_circle_ll_2;
			double theta_circle_rl_1, theta_circle_rl_2, theta_circle_lr_1, theta_circle_lr_2;
			double theta_circle_rr, theta_circle_ll, theta_circle_rl, theta_circle_lr;
			double theta_circle_min, theta_circle[4];
			
			center_lat_start_right = latitude + length1 * _lat * cos(current_yaw_rad) - R1 * _lat * sin(current_yaw_rad);
			center_lon_start_right = longitude + length1 * _lon * sin(current_yaw_rad) + R1 * _lon * cos(current_yaw_rad);
			center_lat_end_right = land_preparing_lat - R2 * _lat * sin(land_yaw_rad);
			center_lon_end_right = land_preparing_lon + R2 * _lon * cos(land_yaw_rad); 
			center_lat_start_left = latitude + length1 * _lat * cos(current_yaw_rad) + R1 * _lat * sin(current_yaw_rad);//怎么这里又变成length1了，到底什么时候用length1什么时候用length2
			center_lon_start_left = longitude + length1 * _lon * sin(current_yaw_rad) - R1 * _lon * cos(current_yaw_rad);
			center_lat_end_left = land_preparing_lat + R2 * _lat * sin(land_yaw_rad);
			center_lon_end_left = land_preparing_lon - R2 * _lon * cos(land_yaw_rad);
		
			center_theta_rr = 0.5 * PI - atan2((center_lat_end_right-center_lat_start_right)/_lat, (center_lon_end_right-center_lon_start_right)/_lon);
			center_theta_rl = 0.5 * PI - atan2((center_lat_end_left-center_lat_start_right)/_lat, (center_lon_end_left-center_lon_start_right)/_lon);
			center_theta_lr = 0.5 * PI - atan2((center_lat_end_right-center_lat_start_left)/_lat, (center_lon_end_right-center_lon_start_left)/_lon);
			center_theta_ll = 0.5 * PI - atan2((center_lat_end_left-center_lat_start_left)/_lat, (center_lon_end_left-center_lon_start_left)/_lon);
		
			center_distance_rl = sqrt(pow((center_lat_start_right-center_lat_end_left)/_lat, 2) + pow((center_lon_start_right-center_lon_end_left)/_lon, 2));
			center_distance_lr = sqrt(pow((center_lat_start_left-center_lat_end_right)/_lat, 2) + pow((center_lon_start_left-center_lon_end_right)/_lon, 2));
		
			theta_circle_rr_1 = - current_yaw_rad + center_theta_rr;
			theta_circle_rr_2 = land_yaw_rad - center_theta_rr;
			theta_circle_ll_1 = current_yaw_rad - center_theta_ll;
			theta_circle_ll_2 = - land_yaw_rad + center_theta_ll;
			theta_circle_rr_1 = theta_circle_rr_1 - ((int)(theta_circle_rr_1/(2*PI)))*2*PI;
			theta_circle_rr_2 = theta_circle_rr_2 - ((int)(theta_circle_rr_2/(2*PI)))*2*PI;
			theta_circle_ll_1 = theta_circle_ll_1 - ((int)(theta_circle_ll_1/(2*PI)))*2*PI;
			theta_circle_ll_2 = theta_circle_ll_2 - ((int)(theta_circle_ll_2/(2*PI)))*2*PI;
			theta_circle_rr_1 = theta_circle_rr_1 >= 0 ? theta_circle_rr_1 : theta_circle_rr_1 + 2*PI;
			theta_circle_rr_2 = theta_circle_rr_2 >= 0 ? theta_circle_rr_2 : theta_circle_rr_2 + 2*PI;
			theta_circle_ll_1 = theta_circle_ll_1 >= 0 ? theta_circle_ll_1 : theta_circle_ll_1 + 2*PI;
			theta_circle_ll_2 = theta_circle_ll_2 >= 0 ? theta_circle_ll_2 : theta_circle_ll_2 + 2*PI;//在计算什么的角度? rr 或则ll模式下 第一段或者第二段圆弧转过的角度
			if(center_distance_rl > R1 + R2) {
				theta_circle_rl_1 = 0.5 * PI - current_yaw_rad + center_theta_rl - acos((R1+R2)/center_distance_rl);//先计算当前航向到正北的角度再计算从正北的弧顶到相切位置的角度
				theta_circle_rl_2 = 0.5 * PI - land_yaw_rad + center_theta_rl - acos((R1+R2)/center_distance_rl);//同理
				theta_circle_rl_1 = theta_circle_rl_1 - ((int)(theta_circle_rl_1/(2*PI)))*2*PI;
				theta_circle_rl_2 = theta_circle_rl_2 - ((int)(theta_circle_rl_2/(2*PI)))*2*PI;
				theta_circle_rl_1 = theta_circle_rl_1 >= 0 ? theta_circle_rl_1 : theta_circle_rl_1 + 2*PI;
				theta_circle_rl_2 = theta_circle_rl_2 >= 0 ? theta_circle_rl_2 : theta_circle_rl_2 + 2*PI;
			}
			else {
				theta_circle_rl_1 = 10000;
				theta_circle_rl_2 = 10000;
			}
			if(center_distance_lr > R1 + R2) {
				theta_circle_lr_1 = 0.5 * PI + current_yaw_rad - center_theta_lr - acos((R1+R2)/center_distance_lr);
				theta_circle_lr_2 = 0.5 * PI + land_yaw_rad - center_theta_lr - acos((R1+R2)/center_distance_lr);
				theta_circle_lr_1 = theta_circle_lr_1 - ((int)(theta_circle_lr_1/(2*PI)))*2*PI;
				theta_circle_lr_2 = theta_circle_lr_2 - ((int)(theta_circle_lr_2/(2*PI)))*2*PI;
				theta_circle_lr_1 = theta_circle_lr_1 >= 0 ? theta_circle_lr_1 : theta_circle_lr_1 + 2*PI;
				theta_circle_lr_2 = theta_circle_lr_2 >= 0 ? theta_circle_lr_2 : theta_circle_lr_2 + 2*PI;
			}
			else {
				theta_circle_lr_1 = 10000;
				theta_circle_lr_2 = 10000;
			}
		
			theta_circle_rr = theta_circle_rr_1 + theta_circle_rr_2;
			theta_circle_ll = theta_circle_ll_1 + theta_circle_ll_2;
			theta_circle_rl = theta_circle_rl_1 + theta_circle_rl_2;
			theta_circle_lr = theta_circle_lr_1 + theta_circle_lr_2;
			theta_circle[0] = theta_circle_rr;
			theta_circle[1] = theta_circle_ll;
			theta_circle[2] = theta_circle_rl;
			theta_circle[3] = theta_circle_lr;
		
			if(Whether_appoint_choice == 0)
			{
				theta_circle_min = theta_circle[0];
				for(int i = 1; i < 4; i++){
					if(theta_circle_min >= theta_circle[i]) {
						theta_circle_min = theta_circle[i];
						choice = i;
					}
				} 
			}
			else if(Whether_appoint_choice == 1)
			{
				choice = Land_choice;
			}

			if(choice == 0)
			{
				j = 0;
				for (double i = 0; i < theta_circle_rr_1 + 0.0000000001; i += theta_circle_rr_1 / (num_waypoints_1-1), j++)
				{
					latP1[j] = center_lat_start_right + R1 * _lat * sin(current_yaw_rad + i);
					lonP1[j] = center_lon_start_right - R1 * _lon * cos(current_yaw_rad + i); 
				}
				count_1 = j;
			
				j = 0;
				for (double i = 0; i < theta_circle_rr_2 + 0.0000000001; i += theta_circle_rr_2 / (num_waypoints_2-1), j++)
				{
					latP2[j] = center_lat_end_right + R2 * _lat * sin(land_yaw_rad - i);
					lonP2[j] = center_lon_end_right - R2 * _lon * cos(land_yaw_rad - i);
				} 
				count_2 = j;
			} 
		
			else if(choice == 1)
			{
				j = 0;
				for (double i = 0; i < theta_circle_ll_1 + 0.0000000001; i += theta_circle_ll_1 / (num_waypoints_1-1), j++)
				{
					latP1[j] = center_lat_start_left - R1 * _lat * sin(current_yaw_rad - i);
					lonP1[j] = center_lon_start_left + R1 * _lon * cos(current_yaw_rad - i); 
				} 
				count_1 = j;
			
				j = 0;
				for (double i = 0; i < theta_circle_ll_2 + 0.0000000001; i += theta_circle_ll_2 / (num_waypoints_2-1), j++)
				{
					latP2[j] = center_lat_end_left - R2 * _lat * sin(land_yaw_rad + i);
					lonP2[j] = center_lon_end_left + R2 * _lon * cos(land_yaw_rad + i);
				} 
				count_2 = j;
			} 
		
			else if(choice == 2)
			{
				j = 0;
				for (double i = 0; i < theta_circle_rl_1 + 0.0000000001; i += theta_circle_rl_1 / (num_waypoints_1-1), j++)
				{
					latP1[j] = center_lat_start_right + R1 * _lat * sin(current_yaw_rad + i);
					lonP1[j] = center_lon_start_right - R1 * _lon * cos(current_yaw_rad + i); 
				} 
				count_1 = j;
			
				j = 0;
				for (double i = 0; i < theta_circle_rl_2 + 0.0000000001; i += theta_circle_rl_2 / (num_waypoints_2-1), j++)
				{
					latP2[j] = center_lat_end_left - R2 * _lat * sin(land_yaw_rad + i);
					lonP2[j] = center_lon_end_left + R2 * _lon * cos(land_yaw_rad + i);
				} 
				count_2 = j;
			} 
		
			else if(choice == 3)
			{
				j = 0;
				for (double i = 0; i < theta_circle_lr_1 + 0.0000000001; i += theta_circle_lr_1 / (num_waypoints_1-1), j++)
				{
					latP1[j] = center_lat_start_left - R1 * _lat * sin(current_yaw_rad - i);
					lonP1[j] = center_lon_start_left + R1 * _lon * cos(current_yaw_rad - i); 
				} 
				count_1 = j;
			
				j = 0;
				for (double i = 0; i < theta_circle_lr_2 + 0.0000000001; i += theta_circle_lr_2 / (num_waypoints_2-1), j++)
				{
					latP2[j] = center_lat_end_right + R2 * _lat * sin(land_yaw_rad - i);
					lonP2[j] = center_lon_end_right - R2 * _lon * cos(land_yaw_rad - i);
				} 
				count_2 = j;
			}

			//0
			w1.frame = 3;
			w1.command = 177;
			w1.param2 = 1.0;
			w1.is_current = false;
			w1.autocontinue = true;
			w1.x_lat = 0;
			w1.y_long = 0;
			w1.z_alt = 0;

			//1
			w.frame = 3;
			w.command = 16;//这里的w来自哪里
			w.param1 = 0;
			w.param2 = 0;
			w.param3 = 0;
			w.param4 = NAN;
			w.is_current = false;
			w.autocontinue = true;
			w.x_lat = latitude;
			w.y_long = longitude;
			w.z_alt = 25.0;
			//request.waypoints.push_back(w);
			request.waypoints.push_back(w);

			//2
			for (int i = 0; i < count_1 && latP1[i] != 0; i++)
			{
				w.x_lat = latP1[i];
				w.y_long = lonP1[i];
				w.z_alt = 25.0 - i;//高度均匀下降，航线按照平面走
				request.waypoints.push_back(w);
			}

			for (int i = 0; i < do_jump_waypoints_num; i++) {
				w1.param1 = 1;
				if(i+num_waypoints_1+2 < 14) w1.param1 = 2+num_waypoints_1+do_jump_waypoints_num;
				request.waypoints.push_back(w1);
			} 

			w2.frame = 3;
			w2.command = 178;
			w2.param1 = 0;
			w2.param3 = 0;

			//3
			for (int i = count_2-1; i >= 0 && latP2[i] != 0; i--)
			{
				w.x_lat = latP2[i];
				w.y_long = lonP2[i];
				w.z_alt = 15;
				request.waypoints.push_back(w);
			}

			//4
			w3.command = 21;
			w3.param1 = 0;
			w3.param2 = 2;//param3没有
			w3.param4 = NAN;
			w3.x_lat = land_lat;
			w3.y_long = land_lon;
			w3.z_alt = 0;
			request.waypoints.push_back(w3);

			waiting = 1;
		}

		if(flag == 1 && waiting == 1)
		{
			cout<<"Landways prepared..."<<endl;
			bool success = land_Client.call(request, response);
			if(success) {
				request.waypoints.clear();
				cout << "Hello, Ground!!!" << endl;
				break;
			}
		}
		ros::spinOnce();
		rate.sleep();
	}
	return 0;
}
