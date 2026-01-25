cmake_minimum_required(VERSION 3.5)

project(legged_rl_description)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY meshes urdf
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
