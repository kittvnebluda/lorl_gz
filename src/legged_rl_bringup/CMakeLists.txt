cmake_minimum_required(VERSION 3.5)

project(legged_rl_bringup)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(legged_rl_description REQUIRED)
find_package(legged_rl_gazebo REQUIRED)

# Install project launch files
install(
  DIRECTORY launch rviz config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
