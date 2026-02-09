#ifndef LEGGED_RL_GAZEBO__RESET_SYSTEM_HH_
#define LEGGED_RL_GAZEBO__RESET_SYSTEM_HH_

#include "legged_rl_interfaces/srv/reset_robot.hpp"
#include <cstdint>
#include <gz/math/Pose3.hh>
#include <gz/sim/Entity.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/service.hpp>
#include <vector>

namespace legged_rl_gazebo {
class ResetSystem
  : public gz::sim::System
  , public gz::sim::ISystemConfigure
  , public gz::sim::ISystemPreUpdate
{
  public:
    ResetSystem();
    ~ResetSystem() override;

  public:
    void Configure(const gz::sim::Entity& _entity,
                   const std::shared_ptr<const sdf::Element>& _element,
                   gz::sim::EntityComponentManager& _ecm,
                   gz::sim::EventManager& _eventManager) override;

    void PreUpdate(const gz::sim::UpdateInfo& _info,
                   gz::sim::EntityComponentManager& _ecm) override;

  private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Service<legged_rl_interfaces::srv::ResetRobot>::SharedPtr service_;

    // Protection for data coming from callback → used in PreUpdate
    std::mutex mutex_;

    bool reset_requested_{ false };
    gz::math::Pose3d desired_pose_;
    std::vector<double> desired_joint_positions_;

    gz::sim::Model model_;
    std::vector<gz::sim::Entity> joint_entities_;
    uint64_t joint_num_;

    void onResetRobot(
      const std::shared_ptr<legged_rl_interfaces::srv::ResetRobot::Request> req,
      std::shared_ptr<legged_rl_interfaces::srv::ResetRobot::Response> res);
};
} // namespace legged_rl_gazebo
#endif
