#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include <gz/math/Pose3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Entity.hh>
#include <gz/sim/components/Pose.hh>
#include <legged_rl_gazebo/PosePublisherSystem.hh>
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>

GZ_ADD_PLUGIN(legged_rl_gazebo::PosePublisherSystem,
              gz::sim::System,
              legged_rl_gazebo::PosePublisherSystem::ISystemConfigure,
              legged_rl_gazebo::PosePublisherSystem::ISystemPostUpdate)

namespace legged_rl_gazebo {
PosePublisherSystem::PosePublisherSystem()
{
    if (!rclcpp::ok())
        rclcpp::init(0, nullptr);

    node_ = rclcpp::Node::make_shared("model_pose_publisher");
};
PosePublisherSystem::~PosePublisherSystem()
{
    if (node_)
        rclcpp::shutdown();
};

void
PosePublisherSystem::Configure(
  const gz::sim::Entity& _entity,
  const std::shared_ptr<const sdf::Element>& _element,
  gz::sim::EntityComponentManager& _ecm,
  gz::sim::EventManager& _eventManager)
{
    model_ = gz::sim::Model(_entity);
    if (!model_.Valid(_ecm)) {
        RCLCPP_ERROR(node_->get_logger(),
                     "PosePublisherSystem must be attached to a model");
        return;
    }

    pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      "real_pose", rclcpp::QoS(10).reliable().transient_local());
    RCLCPP_INFO(node_->get_logger(), "Real pose publisher configured");
}
void
PosePublisherSystem::PostUpdate(const gz::sim::UpdateInfo& _info,
                                const gz::sim::EntityComponentManager& _ecm)
{
    const gz::sim::Entity entity = model_.Entity();
    const auto pose_comp = _ecm.Component<gz::sim::components::Pose>(entity);

    if (!pose_comp) {
        RCLCPP_ERROR(node_->get_logger(), "Model does not have pose component");
        return;
    }

    const gz::math::Pose3d pose = pose_comp->Data();

    auto rot_msg = geometry_msgs::msg::Quaternion();
    auto pose_msg = geometry_msgs::msg::PoseStamped();
    auto pt_msg = geometry_msgs::msg::Point();

    rot_msg.w = pose.Rot().W();
    rot_msg.x = pose.Rot().X();
    rot_msg.y = pose.Rot().Y();
    rot_msg.z = pose.Rot().Z();

    pt_msg.x = pose.X();
    pt_msg.y = pose.Y();
    pt_msg.z = pose.Z();

    pose_msg.header.frame_id = "map";
    pose_msg.header.stamp = node_->now();
    pose_msg.pose.orientation = rot_msg;
    pose_msg.pose.position = pt_msg;

    pose_pub_->publish(pose_msg);
}
}
