cmake_minimum_required(VERSION 3.5)
project(legged_rl_gazebo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(legged_rl_interfaces REQUIRED)

find_package(gz-cmake3 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
set(GZ_PLUGIN_VER ${gz-plugin2_VERSION_MAJOR})
find_package(gz-common5 REQUIRED COMPONENTS profiler)
set(GZ_COMMON_VER ${gz-common5_VERSION_MAJOR})
find_package(gz-sim8 REQUIRED)
set(GZ_SIM_VER ${gz-sim8_VERSION_MAJOR})


add_library(ResetSystem
  SHARED
    src/ResetSystem.cc
)

target_include_directories(ResetSystem
  PRIVATE
    include
)

ament_target_dependencies(ResetSystem
  rclcpp
  legged_rl_interfaces
)

target_link_libraries(ResetSystem
  gz-sim${GZ_SIM_VER}::gz-sim${GZ_SIM_VER}
)


add_library(PosePublisherSystem
  SHARED
    src/PosePublisherSystem.cc
)

target_include_directories(PosePublisherSystem
  PRIVATE
    include
)

ament_target_dependencies(PosePublisherSystem
  rclcpp
  geometry_msgs
)

target_link_libraries(PosePublisherSystem
  gz-sim${GZ_SIM_VER}::gz-sim${GZ_SIM_VER}
)


install(
  TARGETS ResetSystem PosePublisherSystem
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)


if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()


ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.sh.in")

ament_package()
