#include "ros_gz_interfaces/msg/contact.hpp"
#include "ros_gz_interfaces/msg/joint_wrench.hpp"
#include <champ_msgs/msg/contacts_stamped.hpp>
#include <cmath>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ros_gz_interfaces/msg/contacts.hpp>

#include <array>
#include <inttypes.h>
#include <vector>

using namespace std::chrono_literals;

class FootContactAggregator : public rclcpp::Node
{
  public:
    FootContactAggregator()
      : Node("foot_contact_aggregator")
    {
        auto qos = rclcpp::QoS(10).reliable();

        sub_lf_ = create_subscription<ros_gz_interfaces::msg::Contacts>(
          "lf_foot_contacts",
          qos,
          std::bind(&FootContactAggregator::callbackLF, this, std::placeholders::_1));

        sub_rf_ = create_subscription<ros_gz_interfaces::msg::Contacts>(
          "rf_foot_contacts",
          qos,
          std::bind(&FootContactAggregator::callbackRF, this, std::placeholders::_1));

        sub_lh_ = create_subscription<ros_gz_interfaces::msg::Contacts>(
          "lh_foot_contacts",
          qos,
          std::bind(&FootContactAggregator::callbackLH, this, std::placeholders::_1));

        sub_rh_ = create_subscription<ros_gz_interfaces::msg::Contacts>(
          "rh_foot_contacts",
          qos,
          std::bind(&FootContactAggregator::callbackRH, this, std::placeholders::_1));

        pub_ = create_publisher<champ_msgs::msg::ContactsStamped>("foot_contacts", qos);

        timer_ =
          create_wall_timer(10ms, std::bind(&FootContactAggregator::publish, this)); // 100 Hz

        timeout_ = 0.02;
        force_threshold_ = 0.01;

        for (auto& t : last_times_)
            t = now();

        RCLCPP_INFO(this->get_logger(), "Foot contact aggregator started");
    }

  private:
    double norm(const geometry_msgs::msg::Vector3 vec)
    {
        return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    }

    bool has_contact(const ros_gz_interfaces::msg::Contacts::SharedPtr msg)
    {
        if (msg->contacts.empty())
        {
            RCLCPP_INFO(this->get_logger(), "Contacts are empty");
            return false;
        }
        if (msg->contacts.size() > 1)
        {
            RCLCPP_WARN(this->get_logger(), "More than one contact!");
        }
        ros_gz_interfaces::msg::Contact contact = msg->contacts.at(0);

        if (contact.wrenches.size() > 1)
        {
            RCLCPP_WARN(this->get_logger(), "More than one wrench!");
        }
        ros_gz_interfaces::msg::JointWrench wrench = contact.wrenches.at(0);

        double force_norm = norm(wrench.body_1_wrench.force);
        if (force_norm < force_threshold_)
        {
            RCLCPP_DEBUG(
              this->get_logger(), "Force %f below threshold for contact detection", force_norm);
            return false;
        }
        return true;
    }

    void callbackLF(const ros_gz_interfaces::msg::Contacts::SharedPtr msg)
    {
        if (has_contact(msg))
        {
            contacts_[0] = true;
            last_times_[0] = now();
        }
    }

    void callbackRF(const ros_gz_interfaces::msg::Contacts::SharedPtr msg)
    {
        if (has_contact(msg))
        {
            contacts_[1] = true;
            last_times_[1] = now();
        }
    }

    void callbackLH(const ros_gz_interfaces::msg::Contacts::SharedPtr msg)
    {
        if (has_contact(msg))
        {
            contacts_[2] = true;
            last_times_[2] = now();
        }
    }

    void callbackRH(const ros_gz_interfaces::msg::Contacts::SharedPtr msg)
    {
        if (has_contact(msg))
        {
            contacts_[3] = true;
            last_times_[3] = now();
        }
    }

    void publish()
    {
        auto now_time = now();
        for (size_t i = 0; i < 4; ++i)
        {
            if ((now_time - last_times_[i]).seconds() > timeout_)
            {
                contacts_[i] = false;
            }
        }

        auto msg = champ_msgs::msg::ContactsStamped();
        msg.header.stamp = now_time;
        msg.header.frame_id = "base_link";
        msg.contacts = contacts_;
        pub_->publish(msg);
    }

    std::vector<bool> contacts_{ false, false, false, false };
    std::array<rclcpp::Time, 4> last_times_;
    double timeout_;
    double force_threshold_;

    rclcpp::Subscription<ros_gz_interfaces::msg::Contacts>::SharedPtr sub_lf_;
    rclcpp::Subscription<ros_gz_interfaces::msg::Contacts>::SharedPtr sub_rf_;
    rclcpp::Subscription<ros_gz_interfaces::msg::Contacts>::SharedPtr sub_lh_;
    rclcpp::Subscription<ros_gz_interfaces::msg::Contacts>::SharedPtr sub_rh_;

    rclcpp::Publisher<champ_msgs::msg::ContactsStamped>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int
main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FootContactAggregator>());
    rclcpp::shutdown();
    return 0;
}
