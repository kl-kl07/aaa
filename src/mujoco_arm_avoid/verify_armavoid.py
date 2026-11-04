# -*- coding: utf-8 -*-
"""机械臂空间避障轨迹规划验证脚本（对应审阅关注点：IK 数值精度 + 避障安全距离 + 实时渲染）

用法：
    python3 verify_armavoid.py ik     # IK 精度：轨迹抽样 逆解->正解互查
    python3 verify_armavoid.py clear  # 避障安全：全程轨迹与障碍球最小距离
    python3 verify_armavoid.py gui    # glfw 实时渲染：完整播放 15 秒 / 450 帧
"""
import os, math, sys
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco_arm_avoid_single as m


def _qa(model):
    qa = {}
    for name in ('yaw', 'pitch1', 'pitch2'):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name.encode())
        qa[name] = model.jnt_qposadr[jid]
    return qa


def run_ik():
    """轨迹抽样 20 点：ik3 逆解 -> 写入 qpos -> 仿真器正解 -> 对比目标。"""
    model = mujoco.MjModel.from_xml_string(m.MODEL_XML)
    data = mujoco.MjData(model)
    qa = _qa(model)
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b'tip')

    print("== IK 数值精度验证（轨迹抽样 20 点） ==")
    max_err = 0.0
    for k in range(20):
        f = k * (m.TOTAL - 1) // 19
        tx, ty, tz = m.trajectory_point(f)
        q = m.ik3(tx, ty, tz)
        if q is None:
            print("  [FAIL] 点 f=%d 超出工作空间" % f)
            sys.exit(1)
        data.qpos[qa['yaw']] = q[0]
        data.qpos[qa['pitch1']] = q[1]
        data.qpos[qa['pitch2']] = q[2]
        mujoco.mj_forward(model, data)
        p = data.xpos[tip]
        e = math.hypot(p[0] - tx, p[1] - ty, p[2] - tz) * 1000.0
        max_err = max(max_err, e)
        if k % 5 == 0:
            print("  f%3d 目标(%.3f, %.3f, %.3f) -> 正解(%.3f, %.3f, %.3f) 误差 %.4f mm"
                  % (f, tx, ty, tz, p[0], p[1], p[2], e))
    print("最大误差：%.4f mm" % max_err)
    print("结果：" + ("PASS（<0.01 mm）" if max_err < 0.01 else "FAIL"))
    return 0 if max_err < 0.01 else 1


def run_clear():
    """避障安全：全程轨迹（含插值）与障碍球中心最小距离。"""
    model = mujoco.MjModel.from_xml_string(m.MODEL_XML)
    data = mujoco.MjData(model)
    qa = _qa(model)
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b'tip')

    print("== 避障安全距离验证（全程 %d 帧，要求 >= %.2f m） ==" % (m.TOTAL, m.CLEAR_R))
    min_clear = 1e9
    min_frame = -1
    for i in range(m.TOTAL):
        tx, ty, tz = m.trajectory_point(i)
        q = m.ik3(tx, ty, tz)
        if q is None:
            continue
        data.qpos[qa['yaw']] = q[0]
        data.qpos[qa['pitch1']] = q[1]
        data.qpos[qa['pitch2']] = q[2]
        mujoco.mj_forward(model, data)
        p = data.xpos[tip]
        d = math.hypot(p[0] - m.OBS[0], p[1] - m.OBS[1], p[2] - m.OBS[2])
        if d < min_clear:
            min_clear, min_frame = d, i
    print("轨迹-障碍最小距离：%.3f m（frame %d）" % (min_clear, min_frame))
    print("结果：" + ("PASS（>= %.2f m）" % m.CLEAR_R
                     if min_clear >= m.CLEAR_R else "FAIL"))
    return 0 if min_clear >= m.CLEAR_R else 1


def run_gui():
    """glfw 实时渲染：查看器完整播放避障轨迹（450 帧 / 15 秒）。"""
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_string(m.MODEL_XML)
    data = mujoco.MjData(model)
    qa = _qa(model)

    print("== glfw 实时渲染验证（%d 帧 / %.1f 秒） ==" % (m.TOTAL, m.TOTAL / m.FPS))
    print("弹出 MuJoCo 查看器窗口，播放结束后自动关闭。")

    n = 0
    yaw_prev = None
    with mujoco.viewer.launch_passive(model, data) as v:
        for i in range(m.TOTAL):
            tx, ty, tz = m.trajectory_point(i)
            q = m.ik3(tx, ty, tz, yaw_prev)
            if q is None:
                continue
            yaw_prev = q[0]
            data.ctrl[0] = m.KP * (q[0] - data.qpos[qa['yaw']]) - m.KD * data.qvel[0]
            data.ctrl[1] = m.KP * (q[1] - data.qpos[qa['pitch1']]) - m.KD * data.qvel[1]
            data.ctrl[2] = m.KP * (q[2] - data.qpos[qa['pitch2']]) - m.KD * data.qvel[2]
            for _ in range(3):
                mujoco.mj_step(model, data)
            v.sync()
            n += 1

    print("实时渲染播放完成：%d 帧 / %.1f 秒" % (n, n / m.FPS))
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ik"
    if mode == "ik":
        sys.exit(run_ik())
    elif mode == "clear":
        sys.exit(run_clear())
    elif mode == "gui":
        sys.exit(run_gui())
    else:
        print("用法：python3 verify_armavoid.py [ik|clear|gui]")
        sys.exit(1)
