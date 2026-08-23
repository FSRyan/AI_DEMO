"""
全景图旋转脚本
将测试图片 resize 到 960x480 (2:1 equirectangular格式)，
然后在 pitch, roll, yaw 三个轴上进行旋转，生成多角度视图。

Equirectangular 投影模型：
- 图像宽度对应经度 [-π, π] (yaw)
- 图像高度对应纬度 [-π/2, π/2] (pitch)
- 每个像素对应球面上的一个方向向量，通过旋转矩阵变换后重新映射回图像坐标
"""

import numpy as np
from PIL import Image
import os


def euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """
    欧拉角转旋转矩阵 (ZYX顺序: yaw -> pitch -> roll)
    yaw: 绕Y轴旋转 (左右)
    pitch: 绕X轴旋转 (上下)
    roll: 绕Z轴旋转 (倾斜)
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    # 绕Y轴旋转 (yaw)
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # 绕X轴旋转 (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    # 绕Z轴旋转 (roll)
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    # 组合旋转: R = Rz * Rx * Ry
    R = Rz @ Rx @ Ry
    return R


def rotate_equirectangular(img_array, yaw_deg, pitch_deg, roll_deg):
    """
    对 equirectangular 全景图进行旋转。

    原理：对输出图像的每个像素，反向映射找到旋转前对应的源像素位置。
    1. 输出像素 (x, y) -> 球面坐标 (lon, lat)
    2. 球面坐标 -> 3D方向向量
    3. 应用逆旋转矩阵得到源方向向量
    4. 源方向向量 -> 源球面坐标 (lon', lat')
    5. 源球面坐标 -> 源像素坐标 (x', y')
    """
    h, w = img_array.shape[:2]

    # 构建旋转矩阵及其逆（用于反向映射）
    R = euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    R_inv = R.T  # 旋转矩阵的逆等于转置

    # 生成输出图像所有像素的坐标网格
    u = np.arange(w)
    v = np.arange(h)
    u_grid, v_grid = np.meshgrid(u, v)

    # 像素坐标 -> 球面坐标 (经度lon, 纬度lat)
    lon = (u_grid / w - 0.5) * 2 * np.pi   # [-π, π]
    lat = (0.5 - v_grid / h) * np.pi        # [π/2, -π/2] (顶部为正)

    # 球面坐标 -> 3D单位方向向量
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    # 将所有方向向量组成矩阵 (3, H*W)
    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)

    # 应用逆旋转
    xyz_rot = R_inv @ xyz

    # 3D方向向量 -> 球面坐标
    x_rot = xyz_rot[0]
    y_rot = xyz_rot[1]
    z_rot = xyz_rot[2]

    lon_rot = np.arctan2(x_rot, z_rot)  # [-π, π]
    lat_rot = np.arcsin(np.clip(y_rot, -1, 1))  # [-π/2, π/2]

    # 球面坐标 -> 源像素坐标
    u_src = (lon_rot / (2 * np.pi) + 0.5) * w
    v_src = (0.5 - lat_rot / np.pi) * h

    # reshape 回图像尺寸
    u_src = u_src.reshape(h, w)
    v_src = v_src.reshape(h, w)

    # 双线性插值
    output = bilinear_interpolate(img_array, u_src, v_src)
    return output


def bilinear_interpolate(img, u, v):
    """
    对图像进行双线性插值采样。
    u, v: 浮点数坐标数组
    对经度方向(u)进行环绕处理（全景图水平方向是连续的）
    """
    h, w = img.shape[:2]

    # 环绕处理水平方向
    u = u % w
    v = np.clip(v, 0, h - 1)

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = (u0 + 1) % w  # 水平环绕
    v1 = np.clip(v0 + 1, 0, h - 1)

    # 插值权重
    du = u - u0
    dv = v - v0

    if img.ndim == 3:
        du = du[:, :, np.newaxis]
        dv = dv[:, :, np.newaxis]

    # 四个相邻像素
    p00 = img[v0, u0]
    p01 = img[v0, u1]
    p10 = img[v1, u0]
    p11 = img[v1, u1]

    # 双线性插值
    result = (1 - du) * (1 - dv) * p00 + \
             du * (1 - dv) * p01 + \
             (1 - du) * dv * p10 + \
             du * dv * p11

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    # 路径设置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "test.jpg")
    output_dir = os.path.join(script_dir, "test")
    os.makedirs(output_dir, exist_ok=True)

    # 读取并 resize 到 960x480 (2:1 equirectangular)
    img = Image.open(input_path)
    img_resized = img.resize((960, 480), Image.LANCZOS)
    img_resized.save(os.path.join(output_dir, "resized_960x480.jpg"))
    print(f"已将原图 resize 到 960x480，保存到 {output_dir}/resized_960x480.jpg")

    img_array = np.array(img_resized)

    # 定义旋转参数
    pitch_angles = [-90, -60, -30, 30, 60, 90]
    roll_angles = [-90, -60, -30, 30, 60, 90]
    yaw_angles = [-90, 90, 180]

    # Pitch 旋转（绕X轴，上下仰俯）
    print("\n--- Pitch 旋转 ---")
    for pitch in pitch_angles:
        rotated = rotate_equirectangular(img_array, yaw_deg=0, pitch_deg=pitch, roll_deg=0)
        filename = f"pitch_{pitch:+04d}.jpg"
        output_path = os.path.join(output_dir, filename)
        Image.fromarray(rotated).save(output_path)
        print(f"  生成: {filename}")

    # Roll 旋转（绕Z轴，倾斜）
    print("\n--- Roll 旋转 ---")
    for roll in roll_angles:
        rotated = rotate_equirectangular(img_array, yaw_deg=0, pitch_deg=0, roll_deg=roll)
        filename = f"roll_{roll:+04d}.jpg"
        output_path = os.path.join(output_dir, filename)
        Image.fromarray(rotated).save(output_path)
        print(f"  生成: {filename}")

    # Yaw 旋转（绕Y轴，左右平移内容）
    print("\n--- Yaw 旋转 ---")
    for yaw in yaw_angles:
        rotated = rotate_equirectangular(img_array, yaw_deg=yaw, pitch_deg=0, roll_deg=0)
        filename = f"yaw_{yaw:+04d}.jpg"
        output_path = os.path.join(output_dir, filename)
        Image.fromarray(rotated).save(output_path)
        print(f"  生成: {filename}")

    print(f"\n完成！所有旋转图像已保存到: {output_dir}")
    print(f"共生成 {len(pitch_angles) + len(roll_angles) + len(yaw_angles)} 张旋转图 + 1 张 resize 原图")


if __name__ == "__main__":
    main()
