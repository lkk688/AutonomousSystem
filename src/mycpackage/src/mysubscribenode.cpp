#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
//new add for custom msg
#include "my_interfaces/msg/num.hpp" //generated under install/mycpackage/msg/num.hpp

using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node
{
public:
    MinimalSubscriber()
        : Node("minimal_subscriber")
    {
        // subscription_ = this->create_subscription<std_msgs::msg::String>(
        //     "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
        subscription_ = this->create_subscription<my_interfaces::msg::Num>(
            "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    }

private:
    // void topic_callback(const std_msgs::msg::String &msg) const
    // {
    //     RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg.data.c_str());
    // }
    void topic_callback(const my_interfaces::msg::Num &msg) const
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "I heard: '" << msg.num << "'");
    }

    //rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Subscription<my_interfaces::msg::Num>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}