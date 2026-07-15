#include <ros/ros.h>
#include <mavros_msgs/WaypointReached.h>
#include <test/NodeControl.h>
#include <test/results.h>
#include <test/control.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>

using namespace std; 

//需要修改的变量
double latitude = 30.5048068;
double longitude = 120.1103818;
bool round1 = true; //true 第一轮 false 第二轮

// 全局变量
//mutex data_mutex;
int current_wp_ = 0;
int flag_ = 0;
chrono::steady_clock::time_point start;
chrono::steady_clock::time_point end_;
atomic<bool> save_pic_started_{true};
atomic<bool> send_pic_started_{false};
atomic<bool> yolo_started_;
atomic<bool> shutdown_flag_{false};
atomic<bool> timing{false};
chrono::duration<double> gap; //时间间隔

test::results result;
test::control pub_flag;

// 航点回调函数
void waypointCallback(const mavros_msgs::WaypointReached::ConstPtr& msg) 
{
    //lock_guard<mutex> lock(data_mutex);
    current_wp_ = msg->wp_seq;//到哪个航点就切换索引 虽然不知道wp_seq是什么
    //ROS_DEBUG("Waypoint updated: %d", current_wp_);
}

// 结果回调函数
void resultsCallback(const test::results::ConstPtr& msg) 
{
    //lock_guard<mutex> lock(data_mutex);
    result = *msg; //test的宏里面包括什么
    flag_ = result.flag;
    //ROS_DEBUG("Flag updated: %d", flag_);
    cout << "flag=" << flag_ << endl;
}

// 节点控制核心函数
bool controlNodes(const vector<string>& node_names, const string& command, 
                 test::NodeControl::Response* res = nullptr) 
{
    bool overall_success = true;
    string message;

    for (const auto& node : node_names) //for c++简化写法：node_names变量node的值的来源 &取地址符号在取node_names里面变量的值的时候省去复制符号直接调用变量地址 const又同时防止了对变量的更改
    {
        string cmd;
        if (command == "start") 
        {
            cmd = "gnome-terminal --title='" + node + 
                  "' -- bash -c 'rosrun test " + node + "; exec bash'";
            ROS_INFO("Starting node: %s", node.c_str());
        } 
        else 
        {
            cmd = "rosnode kill /" + node;
            ROS_INFO("Stopping node: %s", node.c_str());
        }

        int ret = system(cmd.c_str());//c_str()一种方法，将c++的string类型的字符串转换成c认识的字符串风格
        if (ret != 0) 
        {
            ROS_ERROR("Failed to %s node %s (code: %d)", 
                     command.c_str(), node.c_str(), ret);
            overall_success = false;
            message += " " + node;
        }
    }

    if (res) 
    {
        res->success = overall_success;
        res->message = overall_success ? 
            "Operation succeeded" : "Failed on nodes:" + message;
    }
    return overall_success;//这个函数好像没有对节点进行什么实质性的操作，只是在给予终端一些反馈 其实是因为没有看懂system是调用linux命令行面板的函数 括号里面的是输入到终端的指令
}

bool startYoloProgram()
{
    ROS_INFO("Starting YOLO program...");
        
    if (round1)
    {
    // 使用绝对路径确保可靠性
        const char* cmd = "gnome-terminal --title='YOLO Detection' "
                            "-- bash -c '/home/tx2/Desktop/sh/start_yolo_1.sh; exec bash'";//tx2 给机载电脑的指令
        
        int ret = std::system(cmd);//使用C语言风格给cmd赋值了，但是system函数里面传进去的是指针 c_str就是在把c++的字符串转换为system需要的指针 它的参数就是char* ptr
        if (ret != 0) 
        {
            ROS_ERROR("Failed to start YOLO program (code: %d)", ret);
            return false;
        }
    }
    
    else
    {
        const char* cmd = "gnome-terminal --title='YOLO Detection' "
                            "-- bash -c '/home/tx2/Desktop/sh/start_yolo_2.sh; exec bash'";//不同轮数使用不同的yolo脚本
        
        int ret = std::system(cmd);
        if (ret != 0) 
        {
            ROS_ERROR("Failed to start YOLO program (code: %d)", ret);
            return false;
        }
    }
    return true;
}

//停止函数
bool stopYoloProgram() 
{
    ROS_INFO("Stopping YOLO program...");
    //杀掉 python 推理进程
    int ret1;
    if (round1) ret1 = std::system("pkill -f detect_well_gps.py");//c/c++的用双引号括起来的字符串本质上就是char* 指针
    else ret1 = std::system("pkill -f detect_num_gps.py");
    if (ret1 != 0) //和一般我们程序的写法一样在linux命令行里面假如命令执行成功，那么会返回一个值0
    {
        ROS_WARN("No detect_well.py process matched or kill failed (code: %d)", ret1);
    }
    return true;
}

// 控制服务回调函数
bool controlServiceCallback(test::NodeControl::Request& req,
                           test::NodeControl::Response& res) 
{
    return controlNodes(req.node_names, req.command, &res);
}

// 控制线程函数
void controlThreadFunc(ros::Publisher& pub_flag1,ros::Publisher& pub_gps1) 
{
    ros::Rate rate_(20);

    while (!shutdown_flag_ && ros::ok()) 
    {
        //ros::spinOnce();
	    // 获取当前状态（加锁最小化范围）
        int current_wp, current_flag;
        {
            //lock_guard<mutex> lock(data_mutex);
            current_wp = current_wp_;
            current_flag = flag_;
        }

        // 测试代码（实际使用时注释掉）
        //current_wp = 12;
        //current_wp = 13;
        //current_flag = 1;
        //current_flag = 2;
        //ROS_INFO("Current state: wp=%d, flag=%d", current_wp, current_flag);
	    //pub_flag.flag = 2;

        // 条件判断
        if (current_wp == 12 && current_flag == 0 && save_pic_started_ ) 
        {
            controlNodes({"pic_get_node"}, "stop");
            send_pic_started_ = false;
            save_pic_started_ = false;
        }
        // 全天空端 
        else if (!yolo_started_ && current_flag == 0 && current_wp == 12) 
        {
            if (!timing) //开始计时
            {
                start = chrono::steady_clock::now();
                timing = true;
            }
	        if (startYoloProgram()) 
            {
                yolo_started_ = true;
                ROS_INFO("YOLO program started successfully");
            }
        } 
        else if (current_flag == 1 && current_wp == 13)
        {
            pub_flag.flag=2;
            pub_flag1.publish(pub_flag);
            cout << "send flag success!" << endl;
        }
        else if (pub_flag.flag == 2 && !send_pic_started_) 
        {
			sleep(3);
            controlNodes({"pic_send_sky"}, "start");
            send_pic_started_ = true;
            save_pic_started_ = false;
        }

        end_ = chrono::steady_clock::now();
        gap = end_ - start;
        //超时
        if (timing && gap.count() > 100 && current_flag == 0)
        {
            //判断是否打开yolo,开了的话关闭yolo
            if (yolo_started_) 
            {
                stopYoloProgram();
            } 
            int ret = std::system("pkill -f receive_gps");
            if (ret != 0) 
            {
                ROS_WARN("No detect_well.py process matched or kill failed (code: %d)", ret);
            }
            //发送msg给send_ways_circular
            result.latitude = latitude;
            result.longitude = longitude;
            result.flag = 1;
            pub_gps1.publish(result);
            flag_ = 1;
            current_flag = 1;
        }
	
        rate_.sleep();
    }
}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "mission_controller");
    ros::NodeHandle nh;
    
    // 初始化订阅和服务
    ros::Subscriber wp_reached_sub_ = nh.subscribe("/mavros/mission/reached", 1, 
                                                  waypointCallback);
    ros::Subscriber results_sub_ = nh.subscribe("results", 1, 
                                              resultsCallback);
    ros::Publisher pub_flag1=nh.advertise<test::control>("control",1);
    ros::Publisher pub_gps1=nh.advertise<test::results>("results",10);
    ros::ServiceServer control_service_ = nh.advertiseService("node_control",
                                                            controlServiceCallback);
    
    //老办法
    ros::Rate rate_(20);
    for (int i = 100; ros::ok() && i > 0; --i) 
    {
		ros::spinOnce();
		rate_.sleep();
	}

    // 启动控制线程
    std::thread control_thread(controlThreadFunc,std::ref(pub_flag1),std::ref(pub_gps1));
    //controlThreadFunc(pub_flag1);
    ROS_INFO("Mission Controller initialized");
    
    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();
    
    // 清理资源
    shutdown_flag_ = true;
    if (control_thread.joinable()) 
    {
        control_thread.join();
    }
    
    return 0;
}


