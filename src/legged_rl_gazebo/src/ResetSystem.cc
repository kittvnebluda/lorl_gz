#include "legged_rl_gazebo/ResetSystem.hh"
#include "legged_rl_interfaces/srv/reset_robot.hpp"
#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/JointPositionReset.hh>
#include <gz/sim/components/JointType.hh>
#include <gz/sim/components/JointVelocityReset.hh>
#include <gz/sim/components/Name.hh>
#include <inttypes.h>
#include <memory>
#include <mutex>
#include <rclcpp/create_service.hpp>
#include <rclcpp/executors.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/utilities.hpp>
#include <rmw/qos_profiles.h>
#include <sdf/Joint.hh>

GZ_ADD_PLUGIN(legged_rl_gazebo::ResetSystem,
              gz::sim::System,
              legged_rl_gazebo::ResetSystem::ISystemConfigure,
              legged_rl_gazebo::ResetSystem::ISystemPreUpdate)

namespace legged_rl_gazebo {

ResetSystem::ResetSystem()
{
    if (!rclcpp::ok())
        rclcpp::init(0, nullptr);

    node_ = rclcpp::Node::make_shared("reset_robot_server");
};

ResetSystem::~ResetSystem()
{
    if (node_)
        rclcpp::shutdown();
}

void
ResetSystem::Configure(const gz::sim::Entity& _entity,
                       const std::shared_ptr<const sdf::Element>& _element,
                       gz::sim::EntityComponentManager& _ecm,
                       gz::sim::EventManager& _eventManager)
{
    model_ = gz::sim::Model(_entity);
    if (!model_.Valid(_ecm)) {
        RCLCPP_ERROR(node_->get_logger(),
                     "ResetSystem must be attached to a model");
        return;
    }

    for (const auto& joint_entity : model_.Joints(_ecm)) {
        auto type_comp =
          _ecm.Component<gz::sim::components::JointType>(joint_entity);
        if (!type_comp) {
            RCLCPP_WARN(node_->get_logger(),
                        "Joint %lu missing JointType component - skipping.",
                        joint_entity);
            continue;
        }

        sdf::JointType joint_type = type_comp->Data();

        if (joint_type == sdf::JointType::CONTINUOUS ||
            joint_type == sdf::JointType::REVOLUTE) {

            auto name_comp =
              _ecm.Component<gz::sim::components::Name>(joint_entity);
            if (name_comp) {
                RCLCPP_INFO(node_->get_logger(),
                            "Registered joint: %s (type: %d)",
                            name_comp->Data().c_str(),
                            static_cast<int>(joint_type));
                joint_map_[name_comp->Data()] = joint_entity;
            } else {
                RCLCPP_ERROR(
                  node_->get_logger(),
                  "Registered joint without name component (type: %d)",
                  static_cast<int>(joint_type));
                return;
            }
        } else {
            RCLCPP_DEBUG(node_->get_logger(),
                         "Skipping not supported joint %lu (type: %d)",
                         joint_entity,
                         static_cast<int>(joint_type));
        }
    }

    joint_num_ = joint_map_.size();
    RCLCPP_INFO(
      node_->get_logger(), "Found %zu actuated joints for reset.", joint_num_);

    service_ = node_->create_service<legged_rl_interfaces::srv::ResetRobot>(
      "reset_robot",
      std::bind(&ResetSystem::onResetRobot,
                this,
                std::placeholders::_1,
                std::placeholders::_2),
      rclcpp::QoS(rclcpp::KeepLast(1)));

    RCLCPP_INFO(node_->get_logger(), "Reset robot service created");
}

void
ResetSystem::PreUpdate(const gz::sim::UpdateInfo& _info,
                       gz::sim::EntityComponentManager& _ecm)
{
    rclcpp::spin_some(node_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!reset_requested_)
            return;
        RCLCPP_DEBUG(node_->get_logger(), "Reset requested");

        model_.SetWorldPoseCmd(_ecm, req_pose_);

        for (const auto it : req_joint_map_) {
            std::string joint_name = it.first;
            double joint_pos = it.second;
            auto joint_entity = joint_map_.at(joint_name);

            auto pos_reset =
              _ecm.Component<gz::sim::components::JointPositionReset>(
                joint_entity);
            if (pos_reset) {
                pos_reset->Data() = { joint_pos };
            } else {
                _ecm.CreateComponent(
                  joint_entity,
                  gz::sim::components::JointPositionReset({ joint_pos }));
            }

            auto vel_reset =
              _ecm.Component<gz::sim::components::JointVelocityReset>(
                joint_entity);
            if (vel_reset) {
                vel_reset->Data() = { 0.0 };
            } else {
                _ecm.CreateComponent(
                  joint_entity,
                  gz::sim::components::JointVelocityReset({ 0.0 }));
            }
        }
        reset_requested_ = false;
    }
}

void
ResetSystem::onResetRobot(
  const std::shared_ptr<legged_rl_interfaces::srv::ResetRobot::Request> req,
  std::shared_ptr<legged_rl_interfaces::srv::ResetRobot::Response> res)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (req->joint_positions.size() != req->joint_names.size()) {
        RCLCPP_ERROR(node_->get_logger(),
                     "Reset aborted: joint_positions size does not equal to "
                     "joint_names size. %zu vs %zu.",
                     req->joint_positions.size(),
                     req->joint_names.size());
        res->success = false;
        return;
    }

    for (size_t i = 0; i < req->joint_names.size(); i++)
        req_joint_map_[req->joint_names.at(i)] = req->joint_positions.at(i);

    auto pose = gz::math::Vector3d(
      req->pose.position.x, req->pose.position.y, req->pose.position.z);

    auto quat = gz::math::Quaterniond(req->pose.orientation.w,
                                      req->pose.orientation.x,
                                      req->pose.orientation.y,
                                      req->pose.orientation.z);

    req_pose_.Set(pose, quat);

    reset_requested_ = true;
    res->success = true;
}
} // namespace legged_rl_gazebo
