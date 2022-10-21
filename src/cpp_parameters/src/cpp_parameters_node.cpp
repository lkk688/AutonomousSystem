#include <chrono>
#include <functional>
#include <string>

#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;

class MinimalParam : public rclcpp::Node
{
public:
  MinimalParam()
  : Node("minimal_param_node")
  {
    //Option1: creates a parameter with the name my_parameter and a default value of world
    //The parameter type is inferred from the default value, so in this case it would be set to a string type
    //this->declare_parameter("my_parameter", "world");

    //Option2: with description
    auto param_desc = rcl_interfaces::msg::ParameterDescriptor{};
    param_desc.description = "This parameter is mine!";
    this->declare_parameter("my_parameter", "world", param_desc);

    //the timer_ is initialized with a period of 1000ms, which causes the timer_callback function to be executed once a second.
    timer_ = this->create_wall_timer(
      1000ms, std::bind(&MinimalParam::timer_callback, this));
  }

  void timer_callback()
  {
    //gets the parameter my_parameter from the node, and stores it in my_param
    std::string my_param =
      this->get_parameter("my_parameter").get_parameter_value().get<std::string>();

    //ensures the message is logged
    RCLCPP_INFO(this->get_logger(), "Hello %s!", my_param.c_str());

    //The set_parameters function then sets the parameter my_parameter back to the default string value
    std::vector<rclcpp::Parameter> all_new_parameters{rclcpp::Parameter("my_parameter", "world")};
    this->set_parameters(all_new_parameters);
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalParam>());
  rclcpp::shutdown();
  return 0;
}