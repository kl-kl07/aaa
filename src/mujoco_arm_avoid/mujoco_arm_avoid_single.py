# -*- coding: utf-8 -*-
"""MuJoCo 三自由度机械臂空间避障轨迹规划仿真（单文件，XML 内嵌）

机械臂末端从起点 A 运动到终点 B，途中绕过红色球形障碍物：
路径由避障途经点 W 折线组成（A -> W -> B），解析 3 自由度空间 IK
（肩方位 yaw + 肩俯仰 + 肘俯仰）+ PD 位置控制逐点跟踪，
3/4 视角相机 + 手算 lookAt 透视投影叠加黄色末端轨迹线，
输出 demo_armavoid.gif 效果图。

运行（虚拟机 / 无显卡环境）：
    pip install mujoco pillow numpy
    MUJOCO_GL=osmesa python3 mujoco_arm_avoid_single.py

输出：同目录 demo_armavoid.gif
"""
import os, math

os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import mujoco
from PIL import Image, ImageDraw

# ============ 模型（XML 内嵌） ============
MODEL_XML = r"""
<mujoco model="armavoid">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.6 0.6 0.6" diffuse="0.4 0.4 0.4"/>
  </visual>
  <worldbody>
    <light name="l1" diffuse="0.75 0.75 0.75" pos="0.6 0.6 1.2" dir="-0.4 -0.4 -1" castshadow="false"/>
    <light name="l2" diffuse="0.35 0.35 0.4" pos="-0.8 -0.6 1.0" dir="0.6 0.4 -1" castshadow="false"/>
    <geom name="floor" type="plane" size="1.0 1.0 0.02" rgba="0.88 0.88 0.9 1"/>

    <!-- 三自由度空间机械臂（肩方位 yaw + 肩俯仰 + 肘俯仰） -->
    <body name="base" pos="0 0 0">
      <geom name="basegeom" type="cylinder" size="0.07 0.05" pos="0 0 0.05" rgba="0.32 0.32 0.38 1"/>
      <joint name="yaw" type="hinge" axis="0 0 1" pos="0 0 0.06" range="-180 180"/>
      <body name="shoulder" pos="0 0 0.06">
        <geom name="shoulderjoint" type="sphere" size="0.022" pos="0 0 0" rgba="0.45 0.45 0.5 1"/>
        <joint name="pitch1" type="hinge" axis="0 -1 0" pos="0 0 0" range="-120 120"/>
        <body name="l1" pos="0 0 0">
          <geom name="link1" type="capsule" fromto="0 0 0 0.30 0 0" size="0.035" rgba="0.2 0.42 0.92 1"/>
          <joint name="pitch2" type="hinge" axis="0 -1 0" pos="0.30 0 0" range="-150 150"/>
          <body name="l2" pos="0.30 0 0">
            <geom name="link2" type="capsule" fromto="0 0 0 0.30 0 0" size="0.028" rgba="0.95 0.55 0.12 1"/>
            <body name="tip" pos="0.30 0 0">
              <geom name="tipball" type="sphere" size="0.028" rgba="0.9 0.15 0.15 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- 障碍球（半透明红色） -->
    <geom name="obstacle" type="sphere" size="0.07" pos="0.25 0.05 0.34" rgba="0.9 0.18 0.18 0.45"/>

    <!-- 起点 / 终点 地面标记 -->
    <geom name="mark_a" type="cylinder" size="0.025 0.002" pos="0.20 0.05 0.004" rgba="0.15 0.8 0.2 0.8"/>
    <geom name="mark_b" type="cylinder" size="0.025 0.002" pos="0.30 0.05 0.004" rgba="0.15 0.3 0.9 0.8"/>
  </worldbody>
  <actuator>
    <motor name="m_yaw"   joint="yaw"    ctrlrange="-30 30" ctrllimited="true"/>
    <motor name="m_pitch1" joint="pitch1" ctrlrange="-30 30" ctrllimited="true"/>
    <motor name="m_pitch2" joint="pitch2" ctrlrange="-30 30" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

# ============ 参数 ============
L1 = 0.30
L2 = 0.30
BASE_Z = 0.06            # 肩俯仰中心高度（ik3 中 z 以此为基准）

# 空间关键点（世界坐标）
A = (0.20, 0.05, 0.20)            # 起点
B = (0.30, 0.05, 0.48)            # 终点
W = (0.42, -0.12, 0.36)           # 避障途经点（绕开障碍球外侧）
OBS = (0.25, 0.05, 0.34)          # 障碍球中心
OBS_R = 0.07                      # 障碍球半径
CLEAR_R = 0.10                    # 安全间距（障碍半径 + 末端半径 + 余量）

# 时间线（帧，共 450 帧 / 15 s，FPS=30）
F_W_END = 150        # A -> W 结束
F_B_END = 330        # W -> B 结束（之后停留展示）
TOTAL = 450

FPS = 30
KP = 40.0
KD = 10.0
WIDTH, HEIGHT = 640, 480
FOVY = 45.0
F = (HEIGHT / 2.0) / math.tan(math.radians(FOVY) / 2.0)

# 相机（3/4 视角，与渲染器一致）
CAM_LOOKAT = (0.25, 0.05, 0.30)
CAM_DIST = 1.10
CAM_AZ = 60.0
CAM_EL = -30.0


def ik3(x, y, z, yaw_prev=None):
    """三自由度空间 IK：肩方位 yaw + 臂平面内二连杆（pitch1, pitch2 肘上解）。
    返回 (yaw, q1, q2)；yaw_prev 用于跨 ±pi 解缠。"""
    r = math.hypot(x, y)
    z = z - BASE_Z                       # 相对肩俯仰中心
    d = math.hypot(r, z)
    if d > L1 + L2 - 1e-6 or d < abs(L1 - L2) + 1e-6 or r < 1e-9:
        return None
    yaw = math.atan2(y, x)
    if yaw_prev is not None:
        while yaw - yaw_prev > math.pi:
            yaw -= 2 * math.pi
        while yaw - yaw_prev < -math.pi:
            yaw += 2 * math.pi
    cos_q2 = (d * d - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    q2 = math.acos(max(-1.0, min(1.0, cos_q2)))            # 肘上
    q1 = math.atan2(z, r) - math.atan2(L2 * math.sin(q2),
                                       L1 + L2 * math.cos(q2))
    return yaw, q1, q2


def trajectory_point(frame):
    """按帧返回当前目标点（A -> W -> B 两段线性插值）。"""
    if frame < F_W_END:
        t = frame / F_W_END
        return tuple(A[k] + (W[k] - A[k]) * t for k in range(3))
    if frame < F_B_END:
        t = (frame - F_W_END) / (F_B_END - F_W_END)
        return tuple(W[k] + (B[k] - W[k]) * t for k in range(3))
    return B


def project(x, y, z):
    """手算 lookAt 透视投影（与渲染相机一致）。返回 (u, v) 或 None。"""
    az = math.radians(CAM_AZ)
    el = math.radians(CAM_EL)
    cx, cy, cz = CAM_LOOKAT
    ex = cx + CAM_DIST * math.cos(el) * math.cos(az)
    ey = cy + CAM_DIST * math.cos(el) * math.sin(az)
    ez = cz + CAM_DIST * math.sin(el)

    fwd = np.array([cx - ex, cy - ey, cz - ez])
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    upv = np.cross(right, fwd)

    rel = np.array([x - ex, y - ey, z - ez])
    pz = float(np.dot(rel, fwd))
    if pz <= 1e-4:
        return None
    px = float(np.dot(rel, right))
    py = float(np.dot(rel, upv))
    return (WIDTH / 2 + px / pz * F, HEIGHT / 2 - py / pz * F)


def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    def jq(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name.encode())
        if jid < 0:
            raise RuntimeError('joint %s not found' % name)
        return model.jnt_qposadr[jid], model.jnt_dofadr[jid]

    (qa_y, va_y), (qa_p1, va_p1), (qa_p2, va_p2) = jq('yaw'), jq('pitch1'), jq('pitch2')

    # 渲染器 + 相机
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array(CAM_LOOKAT)
    cam.distance = CAM_DIST
    cam.azimuth = CAM_AZ
    cam.elevation = CAM_EL

    print("空间避障：A=%s -> W=%s -> B=%s，障碍 %s r=%.2f" % (A, W, B, OBS, OBS_R))
    print("安全间距要求：轨迹任意点到障碍中心 >= %.2f m" % CLEAR_R)

    import time
    t0 = time.time()
    frames = []
    traj_px = []            # 已走过的末端轨迹（像素）
    min_clear = 1e9
    yaw_prev = None

    for i in range(TOTAL):
        tx, ty, tz = trajectory_point(i)

        # 避障校验：当前目标与障碍距离
        d_obs = math.hypot(tx - OBS[0], ty - OBS[1], tz - OBS[2])
        if d_obs < CLEAR_R and i < F_B_END:
            print("  [warn] 轨迹点距障碍 %.3f m（< %.2f）frame=%d" % (d_obs, CLEAR_R, i))
        min_clear = min(min_clear, d_obs)

        q = ik3(tx, ty, tz, yaw_prev)
        if q is None:
            print("  [warn] 目标越界 frame=%d (%s)" % (i, (tx, ty, tz)))
            continue
        yaw_prev = q[0]

        data.ctrl[0] = KP * (q[0] - data.qpos[qa_y]) - KD * data.qvel[va_y]
        data.ctrl[1] = KP * (q[1] - data.qpos[qa_p1]) - KD * data.qvel[va_p1]
        data.ctrl[2] = KP * (q[2] - data.qpos[qa_p2]) - KD * data.qvel[va_p2]
        for _ in range(3):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cam)
        rgb = renderer.render().copy()

        tip = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b'tip')]
        pp = project(tip[0], tip[1], tip[2])

        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        if pp is not None:
            traj_px.append(pp)
        if len(traj_px) >= 2:
            draw.line(traj_px, fill=(255, 200, 0), width=3, joint='curve')
        # 当前末端红点
        if pp is not None:
            draw.ellipse([pp[0] - 5, pp[1] - 5, pp[0] + 5, pp[1] + 5],
                         fill=(230, 40, 40))
        frames.append(im)

        if (i + 1) % (TOTAL // 4) == 0:
            print("  已渲染 %d/%d 帧（%.0f%%）" % (i + 1, TOTAL,
                                                  (i + 1) * 100.0 / TOTAL))

    print("轨迹-障碍最小距离：%.3f m（要求 >= %.2f m）" % (min_clear, CLEAR_R))
    print("渲染完成，耗时 %.1f 秒，合成 GIF..." % (time.time() - t0))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_armavoid.gif")
    if frames:
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / FPS), loop=0)
        print("已生成效果图：%s（%d 帧，%.1f 秒）" % (out, len(frames),
                                                  len(frames) / FPS))
    else:
        import sys
        print("未生成任何帧！请检查模型与渲染后端。")
        sys.exit(1)


if __name__ == "__main__":
    main()
