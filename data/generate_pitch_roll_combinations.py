"""
Pitch + Roll 组合旋转脚本
在 pitch 和 roll 各 7 个角度 (-90, -60, -30, 0, 30, 60, 90) 上做组合旋转，
共生成 7x7 = 49 张图，保存到 data/test_pitch_roll/ 目录。
"""

import numpy as np
from PIL import Image
import os


def euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    R = Rz @ Rx @ Ry
    return R


def rotate_equirectangular(img_array, yaw_deg, pitch_deg, roll_deg):
    h, w = img_array.shape[:2]

    R = euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    R_inv = R.T

    u = np.arange(w)
    v = np.arange(h)
    u_grid, v_grid = np.meshgrid(u, v)

    lon = (u_grid / w - 0.5) * 2 * np.pi
    lat = (0.5 - v_grid / h) * np.pi

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)
    xyz_rot = R_inv @ xyz

    x_rot = xyz_rot[0]
    y_rot = xyz_rot[1]
    z_rot = xyz_rot[2]

    lon_rot = np.arctan2(x_rot, z_rot)
    lat_rot = np.arcsin(np.clip(y_rot, -1, 1))

    u_src = (lon_rot / (2 * np.pi) + 0.5) * w
    v_src = (0.5 - lat_rot / np.pi) * h

    u_src = u_src.reshape(h, w)
    v_src = v_src.reshape(h, w)

    output = bilinear_interpolate(img_array, u_src, v_src)
    return output


def bilinear_interpolate(img, u, v):
    h, w = img.shape[:2]

    u = u % w
    v = np.clip(v, 0, h - 1)

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = (u0 + 1) % w
    v1 = np.clip(v0 + 1, 0, h - 1)

    du = u - u0
    dv = v - v0

    if img.ndim == 3:
        du = du[:, :, np.newaxis]
        dv = dv[:, :, np.newaxis]

    p00 = img[v0, u0]
    p01 = img[v0, u1]
    p10 = img[v1, u0]
    p11 = img[v1, u1]

    result = (1 - du) * (1 - dv) * p00 + \
             du * (1 - dv) * p01 + \
             (1 - du) * dv * p10 + \
             du * dv * p11

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "test.jpg")
    output_dir = os.path.join(script_dir, "test_pitch_roll")
    os.makedirs(output_dir, exist_ok=True)

    # 读取并 resize 到 960x480
    img = Image.open(input_path)
    img_resized = img.resize((960, 480), Image.LANCZOS)
    img_array = np.array(img_resized)

    # 7个角度（含0度）
    angles = [-90, -60, -30, 0, 30, 60, 90]

    print(f"Pitch + Roll 组合旋转: {len(angles)}x{len(angles)} = {len(angles)**2} 种组合")
    print(f"输出目录: {output_dir}\n")

    count = 0
    for pitch in angles:
        for roll in angles:
            rotated = rotate_equirectangular(img_array, yaw_deg=0, pitch_deg=pitch, roll_deg=roll)
            filename = f"pitch{pitch:+04d}_roll{roll:+04d}.jpg"
            output_path = os.path.join(output_dir, filename)
            Image.fromarray(rotated).save(output_path)
            count += 1
            print(f"  [{count:2d}/49] {filename}")

    print(f"\n完成！共生成 {count} 张组合旋转图像，保存到: {output_dir}")


if __name__ == "__main__":
    main()
