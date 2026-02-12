import numpy as np
from numpy import float32

# Hips — ±60°
hip_min = -1.0472
hip_max = 1.0472
# Thighs — -90° to +200°
thigh_min = -1.5708
thigh_max = 3.4907
# Calfs — -156° to -48°
calf_min = -2.7227
calf_max = -0.83776

# fmt: off
jpos_low = np.array(
    [
        hip_min, thigh_min, calf_min,  # LF
        hip_min, thigh_min, calf_min,  # RF
        hip_min, thigh_min, calf_min,  # LH
        hip_min, thigh_min, calf_min,  # RH
    ], dtype=float32
)
jpos_high = np.array(
    [
        hip_max, thigh_max, calf_max,
        hip_max, thigh_max, calf_max,
        hip_max, thigh_max, calf_max,
        hip_max, thigh_max, calf_max,
    ], dtype=float32
)
jvel_max = np.array(
    [
        30.1, 30.1, 20.06,  # LF
        30.1, 30.1, 20.06,  # RF
        30.1, 30.1, 20.06,  # LH
        30.1, 30.1, 20.06,  # RH
    ], dtype=float32
)
j_effort_max = np.array(
    [
        23.7, 23.7, 35.55,
        23.7, 23.7, 35.55,
        23.7, 23.7, 35.55,
        23.7, 23.7, 35.55,
    ], dtype=float32
)
# fmt: on

joint_names = (
    "lf_hip_joint",
    "lf_upper_leg_joint",
    "lf_lower_leg_joint",
    "rf_hip_joint",
    "rf_upper_leg_joint",
    "rf_lower_leg_joint",
    "lh_hip_joint",
    "lh_upper_leg_joint",
    "lh_lower_leg_joint",
    "rh_hip_joint",
    "rh_upper_leg_joint",
    "rh_lower_leg_joint",
)

on_tummy_joint_positions = (
    0.8595713850572896,
    0.00241001198312234,
    -1.5517132414045838,
    -0.8586457085662657,
    -0.013527030169100934,
    -1.5577328278092337,
    0.9523385876513574,
    -0.011177661425110218,
    -1.5381480232768159,
    -0.9510537609246051,
    -0.040313159662974184,
    -1.5094152865296782,
)
stable_standing_joint_positions = (
    0.0,
    1.009553554927045,
    -2.0602379537721163,
) * 4
