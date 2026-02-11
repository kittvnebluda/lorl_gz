#include <geometry_msgs/msg/pose_stamped.hpp>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <rclcpp/rclcpp.hpp>

namespace legged_rl_gazebo {
class PosePublisherSystem
  : public gz::sim::System
  , public gz::sim::ISystemConfigure
  , public gz::sim::ISystemPostUpdate
{
  public:
    PosePublisherSystem();
    ~PosePublisherSystem();

  public:
    void Configure(const gz::sim::Entity& _entity,
                   const std::shared_ptr<const sdf::Element>& _element,
                   gz::sim::EntityComponentManager& _ecm,
                   gz::sim::EventManager& _eventManager) override;

    void PostUpdate(const gz::sim::UpdateInfo& _info,
                    const gz::sim::EntityComponentManager& _ecm) override;

  private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    gz::sim::Model model_;
};
}
