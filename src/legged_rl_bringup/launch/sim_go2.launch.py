from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PathSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ros_gz_bridge.actions import RosGzBridge

gz_log_lvls = {"fatal": 0, "error": 1, "warn": 2, "info": 3, "debug": 4}


def generate_launch_description():
    legged_rl_gazebo_pkg_path = FindPackageShare("legged_rl_gazebo")
    legged_rl_bringup_pkg_path = FindPackageShare("legged_rl_bringup")
    legged_rl_description_pkg_path = FindPackageShare("legged_rl_description")
    ros_gz_sim_pkg_path = FindPackageShare("ros_gz_sim")
    unitree_go2_sim_pkg_path = FindPackageShare("unitree_go2_sim")

    default_robot_description = PathJoinSubstitution(
        [legged_rl_description_pkg_path, "urdf", "go2.urdf.xacro"]
    )
    default_gazebo_world = PathJoinSubstitution(
        [legged_rl_gazebo_pkg_path, "worlds", "empty.sdf"]
    )
    default_rviz_cfg = PathJoinSubstitution(
        [legged_rl_bringup_pkg_path, "rviz", "go2.rviz"]
    )
    joints_config = PathJoinSubstitution(
        [unitree_go2_sim_pkg_path, "config", "joints", "joints.yaml"]
    )
    gait_config = PathJoinSubstitution(
        [unitree_go2_sim_pkg_path, "config", "gait", "gait.yaml"]
    )
    links_config = PathJoinSubstitution(
        [unitree_go2_sim_pkg_path, "config", "links", "links.yaml"]
    )
    robot_controllers = PathJoinSubstitution(
        [legged_rl_bringup_pkg_path, "config", "go2_controllers.yaml"]
    )
    ros_gz_bridge_cfg = PathJoinSubstitution(
        [legged_rl_bringup_pkg_path, "config", "go2_bridge.yaml"]
    )
    robot_urdf = Command(
        [
            "xacro ",
            default_robot_description,
            " robot_controllers:=",
            robot_controllers,
            " DEBUG:=",
            LaunchConfiguration("debug"),
            " command_interface:=position",
            " disable_camera:=true",
            " disable_lidar_l1:=true",
            " disable_velodyne_lidar:=true",
        ]
    )
    log_lvl_arg = ["--log-level", LaunchConfiguration("log-level")]
    hostname_user_string = PythonExpression(
        [
            "'",
            EnvironmentVariable("HOSTNAME"),
            "'",
            " + ':' + '",
            EnvironmentVariable("USER", default_value="ayaya"),
            "'",
        ]
    )

    # Declare arguments
    arguments = [
        DeclareLaunchArgument(
            "world",
            default_value=default_gazebo_world,
            description="Absolute SDF world path for the Gazebo to load",
        ),
        DeclareLaunchArgument(
            "debug",
            default_value="false",
            choices=["true", "false"],
            description="If true, the robot will be fixed relative to the map frame",
        ),
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            choices=["true", "false"],
            description="If false, RViz will not be launched, Gazebo will run headless",
        ),
        DeclareLaunchArgument(
            "log-level",
            default_value="warn",
            choices=["fatal", "error", "warn", "info", "debug"],
        ),
        DeclareLaunchArgument(
            "gz-partition",
            default_value=EnvironmentVariable(
                "GZ_PARTITION", default_value=hostname_user_string
            ),
        ),
        DeclareLaunchArgument(
            "ros-domain-id",
            default_value=EnvironmentVariable("ROS_DOMAIN_ID", default_value="52"),
        ),
    ]

    # Set environment
    env = [
        SetEnvironmentVariable("GZ_PARTITION", LaunchConfiguration("gz-partition")),
        SetEnvironmentVariable("ROS_DOMAIN_ID", LaunchConfiguration("ros-domain-id")),
    ]

    # Declare parameters
    parameters = [SetParameter("use_sim_time", True)]

    # Declare nodes
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_urdf}],
        ros_arguments=log_lvl_arg,
    )
    gazebo_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([ros_gz_sim_pkg_path, "launch", "gz_sim.launch.py"])
        ),
        launch_arguments={
            "gz_args": [
                PathSubstitution(LaunchConfiguration("world")),
                PythonExpression(
                    [" ' -s' if '", LaunchConfiguration("gui"), "' == 'false' else ' '"]
                ),
                " -rv",
                PythonExpression(
                    [f"{gz_log_lvls}['", LaunchConfiguration("log-level"), "']"]
                ),
            ],
            "on_exit_shutdown": "True",
        }.items(),
    )
    gazebo_spawn_entity_node = Node(
        package="ros_gz_sim",
        executable="create",
        output="log",
        arguments=[
            "-topic",
            "/robot_description",
            "-z",
            "0.4",
        ],
        ros_arguments=log_lvl_arg,
    )
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="log",
        arguments=["joint_state_broadcaster"],
        ros_arguments=log_lvl_arg,
    )
    delay_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=gazebo_spawn_entity_node,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )
    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="log",
        arguments=[
            "position_controller",
            "--param-file",
            robot_controllers,
        ],
        ros_arguments=log_lvl_arg,
    )
    delay_robot_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robot_controller_spawner],
        )
    )
    initial_joint_positions_publisher = ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "pub",
            "--once",
            "/position_controller/commands",
            "std_msgs/msg/Float64MultiArray",
            "data: [0.0, 1.009553554927045, -2.0602379537721163, "
            "0.0, 1.009553554927045, -2.0602379537721163, "
            "0.0, 1.009553554927045, -2.0602379537721163, "
            "0.0, 1.009553554927045, -2.0602379537721163]",
        ],
        output="log",
    )
    delay_initial_joint_positions_publisher = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_controller_spawner,
            on_exit=[initial_joint_positions_publisher],
        )
    )
    gazebo_bridge = RosGzBridge(
        bridge_name="ros_gazebo_bridge",
        config_file=ros_gz_bridge_cfg,
        log_level=LaunchConfiguration("log-level"),
    )
    # fmt:off
    gazebo_bridge_clock = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[
            {"bridge_names": ["clock_bridge", "real_odometry"]},
            {"bridges.clock_bridge.ros_topic_name": "/clock"},
            {"bridges.clock_bridge.gz_topic_name": "/clock"},
            {"bridges.clock_bridge.ros_type_name": "rosgraph_msgs/msg/Clock"},
            {"bridges.clock_bridge.gz_type_name": "gz.msgs.Clock"},
            {"bridges.clock_bridge.direction": "GZ_TO_ROS"},
            {"bridges.clock_bridge.qos_profile": "CLOCK"},
            {"bridges.real_odometry.ros_topic_name": "/real_odometry"},
            {"bridges.real_odometry.gz_topic_name": "/model/go2/odometry"},
            {"bridges.real_odometry.ros_type_name": "nav_msgs/msg/Odometry"},
            {"bridges.real_odometry.gz_type_name": "gz.msgs.Odometry"},
            {"bridges.real_odometry.direction": "GZ_TO_ROS"},
        ],
    )
    # fmt:on
    foot_contacts_node = Node(
        package="legged_rl_applications",
        executable="foot_contact_aggregator",
        output="log",
        ros_arguments=log_lvl_arg,
    )
    state_estimator_node = Node(
        package="champ_base",
        executable="state_estimation_node",
        output="both",
        parameters=[
            {"orientation_from_imu": True},
            {"urdf": robot_urdf},
            joints_config,
            links_config,
            gait_config,
        ],
        ros_arguments=log_lvl_arg,
        remappings=[("/imu/data", "/imu")],
    )
    base_to_footprint_ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="base_to_footprint_ekf",
        output="log",
        parameters=[
            PathJoinSubstitution(
                [
                    FindPackageShare("champ_base"),
                    "config",
                    "ekf",
                    "base_to_footprint.yaml",
                ]
            ),
        ],
        ros_arguments=log_lvl_arg,
        remappings=[
            ("/imu/data", "/imu"),
            ("/odometry/filtered", "/odom/local"),
            ("/set_pose", "/base_to_footprint_ekf/set_pose"),
        ],
    )
    footprint_to_odom_ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="footprint_to_odom_ekf",
        output="log",
        parameters=[
            PathJoinSubstitution(
                [
                    FindPackageShare("champ_base"),
                    "config",
                    "ekf",
                    "footprint_to_odom.yaml",
                ]
            ),
        ],
        ros_arguments=log_lvl_arg,
        remappings=[
            ("/imu/data", "/imu"),
            ("/odometry/filtered", "/odom"),
            ("/set_pose", "/footprint_to_odom_ekf/set_pose"),
        ],
    )
    # fmt: off
    map_to_odom_tf_node = Node(
        package="tf2_ros",
        name="map_to_odom_tf_node",
        executable="static_transform_publisher",
        output="log",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "0",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "map",
            "--child-frame-id", "odom",
        ],
        ros_arguments=log_lvl_arg,
    )
    # fmt: on
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", default_rviz_cfg],
        ros_arguments=log_lvl_arg,
        condition=IfCondition(LaunchConfiguration("gui")),
    )

    nodes = [
        robot_state_publisher_node,
        gazebo_sim_launch,
        gazebo_bridge,
        gazebo_bridge_clock,
        gazebo_spawn_entity_node,
        delay_joint_state_broadcaster_spawner,
        delay_robot_controller_spawner,
        delay_initial_joint_positions_publisher,
        foot_contacts_node,
        state_estimator_node,
        base_to_footprint_ekf_node,
        footprint_to_odom_ekf_node,
        map_to_odom_tf_node,
        rviz_node,
    ]

    return LaunchDescription(arguments + env + parameters + nodes)
