cmake_minimum_required(VERSION 3.8)
project(legged_rl_applications)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ros_gz_interfaces REQUIRED)
find_package(champ_msgs REQUIRED)

add_executable(foot_contact_aggregator src/foot_contact_aggregator.cpp)
ament_target_dependencies(foot_contact_aggregator
    rclcpp
    ros_gz_interfaces
    champ_msgs)

install(TARGETS
  foot_contact_aggregator
  DESTINATION lib/${PROJECT_NAME})

ament_package()
