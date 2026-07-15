#include <iostream>
#include <cstring>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <ros/ros.h> 
#include <iomanip>
#include <sensor_msgs/NavSatFix.h>
#include <mavros_msgs/WaypointPush.h>
#include <mavros_msgs/GlobalPositionTarget.h>
#include "test/results.h" 
#include "test/control.h" 
using namespace std; 
int Whether_use_8 = 0;
int Whether_use_current = 0;
int Whether_use_control = 1;
double latitude = 30.5048068;
double longitude = 120.1103818;
double circle_latitude = 30.5043030;
double circle_longitude = 120.1089509;
double current_latitude = 30.5041495;
double current_longitude = 120.1090540;
const double R1 = 15;
double R2 = 35;
const double length = 20;
double _lat = 0.0000091256386;
double _lon = 0.00001180199;
const double PI = 3.1415926535897932384626433832795;
const double Earth_Radius = 6367967.0;
const double latitude_now = (2 * latitude + current_latitude) / 3;
const int num_waypoints_1 = 9;
const int num_waypoints_2 = 7;
int flag = 0;
int flag_send = 0;

void metersToLatLonWithAltitude(double &_lat, double &_lon) {
    // 纬度变化（每米对应的纬度差）
    _lat = 180.0 / (Earth_Radius * PI);

    // 经度变化（每米对应的经度差）
    _lon = 180.0 / (Earth_Radius * std::cos(latitude_now * PI / 180.0) * PI);
}

test::results result;
void receive(const test::results::ConstPtr& msg)
{
	result = *msg;

	latitude = result.latitude;
	longitude = result.longitude;
	if(Whether_use_control == 0) {
		flag = result.flag;
	}
}

test::control ctrl;
void ctrl_receive(const test::control::ConstPtr& msg)
{
	ctrl = *msg;

	if(Whether_use_control == 1) {
		flag = ctrl.flag;
	}
}

int gps_flag = 0;
sensor_msgs::NavSatFix global_pos;
void gp_cb(const sensor_msgs::NavSatFix::ConstPtr& msg) {
	if (!msg) {
        ROS_ERROR("Null pointer in GPS callback!");
        return;
    }
	global_pos = *msg;
	if((flag == 1 || flag == 2) && gps_flag == 0) {
        if(Whether_use_current == 1) {
            current_latitude = global_pos.latitude;
            current_longitude = global_pos.longitude;
        }
		gps_flag = 1;
	}
}

int main(int argc, char* argv[]) {
	ros::init(argc, argv, "node");
	ros::NodeHandle nh;
	ros::Rate rate(20.0);  //更新速度20HZ

    mavros_msgs::WaypointPush::Request request;
	mavros_msgs::WaypointPush::Response response;
    mavros_msgs::GlobalPositionTarget pose_tar;
    mavros_msgs::Waypoint w,w1,w2;
	ros::ServiceClient point_Client = nh.serviceClient<mavros_msgs::WaypointPush>("mavros/mission/push");
	ros::Subscriber person_info_sub = nh.subscribe<test::results>("results", 10, receive);
	ros::Publisher global_pos_pub = nh.advertise<mavros_msgs::GlobalPositionTarget>("mavros/setpoint_raw/global", 10);
	ros::Subscriber gp_sub = nh.subscribe<sensor_msgs::NavSatFix>("mavros/global_position/global", 1, gp_cb);
	ros::Subscriber ctrl_sub = nh.subscribe<test::control>("control", 10, ctrl_receive);

	cout<<"Waiting for water-throwing ways generated."<<endl;

    for (int i = 100; ros::ok() && i > 0; --i) {
		ros::spinOnce();
		rate.sleep();
	}//预先使用老办法消耗100/20=5s的时间，使得ros的callback能够真正启动并创建数据

	int waiting = 0;
	while (ros::ok())
	{
		if(gps_flag == 1 && waiting == 0)
		{
            double theta_rad;
            double lat_circle_2, lon_circle_2;
            double distance = 70.0;
            double latP1[100] = { 0 }, lonP1[100] = { 0 }, latP2[100] = { 0 }, lonP2[100] = { 0 };
			metersToLatLonWithAltitude(_lat, _lon);
            if(Whether_use_8 == 1)
            {
                double center_north_lat, center_north_lon, center_south_lat, center_south_lon;
                double theta_circle, theta_begin;
                double theta = 140.0, R = 25, length_whole = 170;
                theta_rad = theta * PI / 180;
                center_north_lat = latitude + length_whole/2 * _lat * cos(theta_rad);//这个对应什么样的图没有看懂
                center_north_lon = longitude + length_whole/2 * _lon * sin(theta_rad);
                center_south_lat = latitude - length_whole/2 * _lat * cos(theta_rad);
                center_south_lon = longitude - length_whole/2 * _lon * sin(theta_rad);
                theta_begin = acos(2*R / length_whole);
                theta_circle = 2*PI - 2 * theta_begin;
                
                int j = 0;
                for (double i = 0; i < theta_circle + 0.0000000001; i += theta_circle / (num_waypoints_1-1), j++)
                {
                    latP1[j] = center_north_lat - R * _lat * cos(theta_rad + theta_begin + i);
                    lonP1[j] = center_north_lon - R * _lon * sin(theta_rad + theta_begin + i); 
                }
                
                j = 0;
                for (double i = 0; i < theta_circle + 0.0000000001; i += theta_circle / (num_waypoints_2-1), j++)
                {
                    latP2[j] = center_south_lat + R * _lat * cos(-theta_rad + theta_begin + i);
                    lonP2[j] = center_south_lon - R * _lon * sin(-theta_rad + theta_begin + i);//沿着圆弧生成航线
                }
            }
            else if(Whether_use_8 == 0)
            {
                R2 = sqrt(pow((circle_latitude-current_latitude)/_lat, 2) + pow((circle_longitude-current_longitude)/_lon, 2));
                double theta_pre_rad = 0.5*PI - atan2((latitude - circle_latitude)/_lat, (longitude - circle_longitude)/_lon);
                double distanceTocenter = sqrt(pow((circle_latitude-latitude)/_lat, 2) + pow((circle_longitude-longitude)/_lon, 2));
                double theta_relative_rad = asin(R2/distanceTocenter);
                theta_rad = theta_pre_rad + theta_relative_rad;
                double center_1_lat = latitude + length * _lat * cos(theta_rad) - R1 * _lat * sin(theta_rad);
                double center_1_lon = longitude + length * _lon * sin(theta_rad) + R1 * _lon * cos(theta_rad);
                double center_2_lat = circle_latitude;
                double center_2_lon = circle_longitude;
                double theta_center_rad = 0.5*PI - atan2((center_1_lat - center_2_lat)/_lat, (center_1_lon - center_2_lon)/_lon);
                
                int j = 0;
                for (double i = 0; i < PI + 0.0000000001; i += PI / (num_waypoints_1-1), j++)
                {
                    latP1[j] = center_1_lat + R1 * _lat * sin(theta_center_rad + i);
                    lonP1[j] = center_1_lon - R1 * _lon * cos(theta_center_rad + i); 
                }
                
                j = 0;
                for (double i = 0; i < PI + 0.0000000001; i += PI / (num_waypoints_2-1), j++)
                {
                    latP2[j] = center_2_lat - R2 * _lat * sin(theta_center_rad + i);
                    lonP2[j] = center_2_lon + R2 * _lon * cos(theta_center_rad + i); 
                    
                    lat_circle_2 = latP2[j];
                    lon_circle_2 = lonP2[j];
                }
                
                distance = sqrt(pow((latitude-lat_circle_2)/_lat, 2) + pow((longitude-lon_circle_2)/_lon, 2));
                distance = (int)(distance / 10) * 10;
            }

            //0	
            w.frame = 3;
            w.command = 16;
            w.param1 = 0;
            w.param2 = 0;
            w.param3 = 0;
            w.param4 = 0;
            w.is_current = false;
            w.autocontinue = true;
            w.x_lat = 0;
            w.y_long = 0;
            w.z_alt = 100;
            request.waypoints.push_back(w);

            w2.frame = 3;
            w2.command = 178;
            w2.param1 = 0;
            w2.param3 = 0;

            //1
            for(int i = 0; latP1[i] != 0; i++)
            {
                w.x_lat = latP1[i];
                w.y_long = lonP1[i];
                w.z_alt = 25.0;
                request.waypoints.push_back(w);
            }

            //2
            if(Whether_use_8 == 1)
            {
                w.x_lat = latitude;
                w.y_long = longitude;
                w.z_alt = 25.0;
                request.waypoints.push_back(w);	
            }

            //3
            for (int i = 0; latP2[i] != 0; i++)
            {
                w.x_lat = latP2[i];
                w.y_long = lonP2[i];
                w.z_alt = 25.0;
                request.waypoints.push_back(w);
            }

            //4
            for(int i = 0; Whether_use_8 == 0 && distance - i * 15 >= 15; i++)
            {
                w.x_lat = latitude - (distance - i * 15) * _lat * cos(theta_rad);
                w.y_long = longitude - (distance - i * 15) * _lon * sin(theta_rad);
                w.z_alt = 25.0;
                request.waypoints.push_back(w);
            }

            //5
            w.x_lat = latitude;
            w.y_long = longitude;
            w.z_alt = 25.0;
            request.waypoints.push_back(w);	

            //6
            w1.frame = 3;
            w1.command = 177;
            w1.param1 = 1.0;
            w1.param2 = 20.0;
            w1.is_current = false;
            w1.autocontinue = true;
            w1.x_lat = 0;
            w1.y_long = 0;
            w1.z_alt = 0;
            request.waypoints.push_back(w1);

            waiting = 1;
		}

		if((flag == 1 || flag == 2) && waiting == 1)
		{
			cout<<"Water-throwing ways prepared..."<<endl;
			bool success = point_Client.call(request, response);
			if(success) {
				waiting = 2;
				flag_send = 1;
				request.waypoints.clear();
				cout<<"Attention!!! Throwing Water!!!"<<endl;
			}
		}

		pose_tar.type_mask=0b1111111111111111;
		pose_tar.latitude = latitude;
		pose_tar.longitude = longitude;
		pose_tar.altitude = flag_send;
		global_pos_pub.publish(pose_tar);//逻辑和landways是一样的依然中间那个center_lat和center_lon的图片的关系是什么样子的

		ros::spinOnce();  //监控回调函数
		rate.sleep(); 
	}
	return 0;
}
