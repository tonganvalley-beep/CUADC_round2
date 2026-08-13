#include<ros/ros.h>
#include<vector>
#include<thread>
#include<atomic>            //加密
#include<chrono>            //时间
#include<unistd.h>
#include<mavros_msgs/RCIn.h>
#include "test/yoloresults.h"
#include "test/sendpicresults.h"
#include "test/throwwaysresults.h"
#include "test/landwaysresults.h"
#include <mavros_msgs/WaypointReached.h>
#include <yaml-cpp/yaml.h>
#include <fstream>

using namespace std;

bool round1 = true;
bool ground_test = false;

int YOLO_FLAG = 0;
int SEND_PIC_FLAG = 0;
int THROW_FLAG = 0;
int LAND_FLAG = 0;
int COMPLETE_FLAG = 0;

const int delay_yolo = 360;          //YOLO延时阈值
const int delay_throw_ways = 180;   //投水任务延时阈值
const int delay_wp = 120;           //投水航点延时阈值

atomic<bool> shutdown_flag{false};  //整个代码关闭

// 航点序号
std::atomic<int> current_wp{-1};
int throw_wp = 0;

// 航点到达回调
void wpReachedCallback(const mavros_msgs::WaypointReached::ConstPtr& msg) {
    if (msg) {
        current_wp = msg->wp_seq;
        ROS_INFO( "Waypoint reached: %d", current_wp.load());
    }
}

//遥控器输入回调函数
mavros_msgs::RCIn rcin_msg;
int land_flag = 0;
void rcinCallback(const mavros_msgs::RCIn::ConstPtr& msg){
    if(!msg){
        ROS_ERROR("Null pointer received in rcinCallback!");
        return;
    }
    rcin_msg = *msg;
    if(land_flag == 0 && rcin_msg.channels.size() > 10 && rcin_msg.channels[10] > 1800){
        land_flag = 1;      //11通，先保证有11通道输入
    }
}

//YOLO检测结果回调函数
test::yoloresults yolo_results_msg;
std::atomic<int> control_yolo_flag{0};          //0未完成 1已完成
void yoloResultsCallback(const test::yoloresults::ConstPtr& msg){
    yolo_results_msg = *msg;
    control_yolo_flag = yolo_results_msg.flag;
}

//发送图片结果回调函数
test::sendpicresults send_pic_results_msg;
std::atomic<int> control_send_pic_flag{0};      //0未完成 1已完成
void sendPicResultsCallback(const test::sendpicresults::ConstPtr& msg){
    send_pic_results_msg = *msg;
    control_send_pic_flag = send_pic_results_msg.flag;
}

//投水任务结果回调函数
test::throwwaysresults throwways_results_msg;
std::atomic<int> control_throwways_flag{0};     //0未完成 1已完成
void throwWaysResultsCallback(const test::throwwaysresults::ConstPtr& msg){
    throwways_results_msg = *msg;
    control_throwways_flag = throwways_results_msg.flag;
}

//降落任务结果回调函数
test::landwaysresults landways_results_msg;
std::atomic<int> control_landways_flag{0};      //0未完成 1已完成
void landWaysResultsCallback(const test::landwaysresults::ConstPtr& msg){
    landways_results_msg = *msg;
    control_landways_flag = landways_results_msg.flag;
}

//节点控制函数
bool controlNodes(const string& node_name, const string& command){
    bool success = true;
    string cmd;
    if(command == "start"){
        if(node_name == "yolo_program"){
            if (round1) {
                    cmd = "gnome-terminal --title='YOLO Detection' "
                        "-- bash -c '/home/tx2/Desktop/sh/start_yolo_1.sh; exec bash'";
            } else {
                    cmd = "gnome-terminal --title='YOLO Detection' "
                        "-- bash -c '/home/tx2/Desktop/sh/start_yolo_2.sh; exec bash'";
            }
        }
        else{
            cmd = "gnome-terminal --title='" + node_name + 
                "' -- bash -c 'rosrun test " + node_name + "; exec bash'";
        }
    } else if(command == "stop"){
        if(node_name == "yolo_program"){
            if(round1){
                cmd = "pkill -f detect_well.py";
            } else {
                cmd = "pkill -f detect_ground.py";
            }
        }
        else{
            cmd = "pkill -INT " + node_name + " 2>/dev/null";
        }
    }

    int ret = system(cmd.c_str());
    if (ret != 0) {
        //ROS_ERROR("Failed to ", command, " node: ", node_name);
        success = false;
    }
    return success;
}


//控制线程函数
void controlThreadFunc(){
    ros::Rate loop_rate(20);

    enum MissionPhase{
        STARTING_VISION_MISSION,        //启动视觉任务，包括拍照、YOLO检测和接收坐标
        WAITING_YOLO_RESULT,            //等待视觉结果
        STARTING_THROWWAYS,             //启动投水
        STARTING_SEND_PIC,              //发送图片
        STARTING_LANDWAYS,              //启动降落
        WAITING_LANDWAYS,               //等待降落
        MISSION_COMPLETE                //所有任务完成
    };

    MissionPhase phase = STARTING_VISION_MISSION;

    //超时开始点，在case前定义
    auto yolo_start_time = chrono::steady_clock::now();
    auto throw_ways_start_time = chrono::steady_clock::now();
    auto wp_start_time = chrono::steady_clock::now();

    if (ground_test) {
        ROS_INFO("======== ground ========");
    } else {
        ROS_INFO("======== sky ========");
    }

    while(!shutdown_flag && ros::ok()){

        //先拷一份防止逻辑中途修改
        int yolo_flag = control_yolo_flag.load();
        int send_pic_flag = control_send_pic_flag.load();
        int throwways_flag = control_throwways_flag.load();
        int landways_flag = control_landways_flag.load();

        // 关闭发送图片 - 全局优先处理
        // 保证发完再关
        if(send_pic_flag == 1 && phase == MISSION_COMPLETE) {
            controlNodes("send_pic", "stop");
            SEND_PIC_FLAG = 0;
            loop_rate.sleep();
            continue;
        }

        // 控切降落 - 全局优先处理
        if(land_flag == 1 && LAND_FLAG == 0) {
            controlNodes("save_pic", "stop");
            controlNodes("throw_ways", "stop");
            controlNodes("receive_gps", "stop");
            if(controlNodes("landways", "start")){
                cout<<"RCIn landing!!!"<<endl;
            }
            loop_rate.sleep();
            break;
        }

        switch(phase){
            // 开始，启动YOLO、拍照、接收坐标
            case STARTING_VISION_MISSION:{
                if(controlNodes("yolo_program", "start")){
                    yolo_start_time = chrono::steady_clock::now();
                    YOLO_FLAG = 1;
                    phase = WAITING_YOLO_RESULT;
                    cout<<"vision mission start, waiting yolo..."<<endl;
                }
                break;
            } 

            // 等待YOLO结果，正常自己关
            case WAITING_YOLO_RESULT:{
                auto yolo_time = chrono::steady_clock::now() - yolo_start_time;
                if(yolo_flag == 1){
                    YOLO_FLAG = 0;
                    phase = STARTING_THROWWAYS;
                    if(!ground_test) wp_start_time = chrono::steady_clock::now();
                    cout<<"yolo complete,throw water..."<<endl;
                }
                else if (chrono::duration_cast<chrono::seconds>(yolo_time).count() > delay_yolo) {
                    controlNodes("yolo_program", "stop");
                    YOLO_FLAG = 0;
                    phase = STARTING_THROWWAYS;
                    if(!ground_test) wp_start_time = chrono::steady_clock::now();
                    cout<<"yolo timeout,throw water!!!"<<endl;
                }
                break;
            }

            // 开始投水
            case STARTING_THROWWAYS:{
                if(ground_test){
                    if (controlNodes("throw_ways", "start")) {
                        throw_ways_start_time = chrono::steady_clock::now();
                        THROW_FLAG = 1;
                        phase = STARTING_SEND_PIC;
                        cout<<"(ground)throw water start..."<<endl;
                    }
                }
                else{
                    auto wp_time = chrono::steady_clock::now() - wp_start_time;
                    if(current_wp.load() == throw_wp || chrono::duration_cast<chrono::seconds>(wp_time).count() > delay_wp){
                        if(current_wp.load() == throw_wp) cout<<"reach throw wp"<<endl;
                        else cout<<"wp timeout!!!"<<endl;
                        if (controlNodes("throw_ways", "start")) {
                            throw_ways_start_time = chrono::steady_clock::now();
					        THROW_FLAG = 1;
                            phase = STARTING_SEND_PIC;
                            cout<<"throw water start..."<<endl;
                        }
                    }
                }
                break;
            }

            // 开始发送图片
            case STARTING_SEND_PIC:{
                if(send_pic_flag == 0){
                    if (controlNodes("send_pic", "start")) {
					SEND_PIC_FLAG = 1;
                    phase = STARTING_LANDWAYS;
                    cout<<"sending to ground..."<<endl;
                    }
                }
                break;
            }

            // 开始降落，控切在上面
            case STARTING_LANDWAYS:{
                auto throw_ways_time = chrono::steady_clock::now() - throw_ways_start_time;
                if(throwways_flag == 1 || chrono::duration_cast<chrono::seconds>(throw_ways_time).count() > delay_throw_ways){
                    controlNodes("throw_ways", "stop");
                    controlNodes("receive_gps", "stop");
                    THROW_FLAG = 0;
                    RECEIVE_GPS_FLAG = 0;
                    SEND_PIC_FLAG = 0;
                    if (controlNodes("landways", "start")) {
						LAND_FLAG = 1;
                        phase = WAITING_LANDWAYS;
                        if(throwways_flag == 1){
                            cout<<"throw water complete,landing..."<<endl;
                        }
                        else{
                            cout<<"throw water timeout,landing!!!"<<endl;
                    	}
                	}
                	break;
            	}
			}

            // 等待降落完成
            case WAITING_LANDWAYS:{
                if(landways_flag == 1){     // 来自landways，代表降落航线已发送
                    controlNodes("landways", "stop"); 
					LAND_FLAG = 0;
					phase = MISSION_COMPLETE;
                    cout<<"land complete!!!"<<endl;
                }
                break;
            }

            case MISSION_COMPLETE:{
                if(YOLO_FLAG == 0 && THROW_FLAG == 0 && SEND_PIC_FLAG == 0 && LAND_FLAG == 0){
                    if(COMPLETE_FLAG == 0){
			        ROS_INFO("======Mission Complete!!!======");
			        COMPLETE_FLAG = 1;
			        }
                }
                break;
            }

            default:
                break;
        }

        loop_rate.sleep();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "main_ctl");
    ros::NodeHandle n;

    // --- 加载配置文件 ---
    std::string config_path = "/home/tx2/Wintter/src/test/config.yaml";
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
        ROS_INFO("config loaded: %s", config_path.c_str());
    } catch (const YAML::Exception& e) {
        ROS_FATAL("config failed: %s  error: %s", config_path.c_str(), e.what());
        return -1;
    }
    ground_test = config["ground_test"].as<bool>();
    throw_wp = config["switch_wp"].as<int>();

    ros::Subscriber rcin_sub = n.subscribe("/mavros/rc/in", 1, rcinCallback);
    ros::Subscriber yoloresults_sub = n.subscribe("yoloresults", 1, yoloResultsCallback);
    ros::Subscriber savepicresults_sub = n.subscribe("savepicresults", 1, savePicResultsCallback);
    ros::Subscriber sendpicresults_sub = n.subscribe("sendpicresults", 1, sendPicResultsCallback);
    ros::Subscriber throwwaysresults_sub = n.subscribe("throwwaysresults", 1, throwWaysResultsCallback);
    ros::Subscriber landwaysresults_sub = n.subscribe("landwaysresults", 1, landWaysResultsCallback);
    ros::Subscriber wp_sub = n.subscribe("/mavros/mission/reached", 1, wpReachedCallback);

    thread control_thread([](){
        controlThreadFunc();
    });

    ros::MultiThreadedSpinner spinner(2);   //2线程处理
    spinner.spin(); //阻塞直到ROS关闭

    shutdown_flag = true;
    if(control_thread.joinable()){
        control_thread.join();
    }

    return 0;
}
