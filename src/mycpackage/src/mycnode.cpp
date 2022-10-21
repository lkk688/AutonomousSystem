#include <cstdio>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

// These lines represent the node’s dependencies, have to be added to package.xml and CMakeLists.txt
#include "rclcpp/rclcpp.hpp"       //allows you to use the most common pieces of the ROS 2 system
#include "std_msgs/msg/string.hpp" //includes the built-in message type you will use to publish data.

//new add for custom msg
#include "my_interfaces/msg/num.hpp" //generated under install/mycpackage/msg/num.hpp

using namespace std::chrono_literals; // provided by the standard library

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class MinimalPublisher : public rclcpp::Node
{
public:
  //The public constructor names the node minimal_publisher and initializes count_ to 0
  MinimalPublisher()
      : Node("minimal_publisher"), count_(0)
  {
    //initialized with the String message type, the topic name topic, and the required queue size to limit messages in the event of a backup.
    //publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    //Use custom msg
    publisher_ = this->create_publisher<my_interfaces::msg::Num>("topic", 10);

    //executed twice a second
    timer_ = this->create_wall_timer(
        500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    //auto message = std_msgs::msg::String();
    auto message = my_interfaces::msg::Num();

    //message.data = "Hello, world! " + std::to_string(count_++);
    message.num = this->count_++

    //The RCLCPP_INFO macro ensures every published message is printed 
    //RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    //Use RCLCPP_INFO for printf-style logging, and RCLCPP_INFO_STREAM for cout-style logging
    RCLCPP_INFO_STREAM(this->get_logger(), "Publishing: '" << message.num << "'"); 

    publisher_->publish(message);
  }

  //the declaration of the timer, publisher, and counter fields.
  rclcpp::TimerBase::SharedPtr timer_;
  
  //rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Publisher<my_interfaces::msg::Num>::SharedPtr publisher_;

  size_t count_;
};

int main(int argc, char **argv)
{
  // (void)argc;
  // (void)argv;
  rclcpp::init(argc, argv); //initializes ROS 2
  //rclcpp::spin starts processing data from the node, including callbacks from the timer.
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();

  printf("hello world mycpackage package\n");
  return 0;
}
