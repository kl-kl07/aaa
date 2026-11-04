# 项目 1：机械臂空间避障轨迹规划（MuJoCo 仿真）

三自由度空间机械臂（肩方位 + 肩俯仰 + 肘俯仰）末端从起点 A 运动到终点 B，
途中绕过半透明红色球形障碍物：路径由避障途经点 W 折线组成（A → W → B），
解析空间 IK + PD 位置控制逐点跟踪，3/4 视角相机渲染，
叠加黄色末端轨迹线，输出 `demo_armavoid.gif` 效果图。

## 文件

| 文件 | 说明 |
|---|---|
| `mujoco_arm_avoid_single.py` | 主程序（单文件，模型 XML 内嵌） |
| `verify_armavoid.py` | 验证脚本（IK 精度 / 避障安全距离 / glfw 实时渲染三种模式） |
| `README.md` | 本项目说明 |
| `docs/mujoco_arm_avoid.md` | 设计与验证文档（IK 互查、避障距离） |
| `docs/mujoco_arm_demo.gif` | 效果图（避障轨迹动画，450 帧 / 15 秒） |

## 运行

```bash
# 本机（Windows glfw 后端）
python mujoco_arm_avoid_single.py

# 虚拟机 / 无显卡环境（osmesa 离屏渲染）
MUJOCO_GL=osmesa python3 mujoco_arm_avoid_single.py
```

运行结束在同目录生成效果图 `demo_armavoid.gif`（450 帧 / 15.0 秒）。

## 验证

```bash
python3 verify_armavoid.py ik      # IK 精度：轨迹抽样 20 点，逆解->正解互查
python3 verify_armavoid.py clear   # 避障安全：全程轨迹与障碍球最小距离
python3 verify_armavoid.py gui     # glfw 实时渲染：完整播放避障轨迹
```

本机验证结果：IK 最大误差 **0.0000 mm**；全程轨迹与障碍球中心最小距离
**0.106 m**（安全间距要求 ≥ 0.10 m，PASS）。

## 设计要点

- 三自由度空间臂：肩方位 yaw + 肩俯仰 pitch1 + 肘俯仰 pitch2，
  杆长 L1 = L2 = 0.30 m，肩俯仰中心高度 0.06 m
- 空间 IK：方位角 `yaw = atan2(y, x)` + 臂平面内二连杆肘上解（带跨 ±π 解缠）
- 任务点：起点 A (0.20, 0.05, 0.20)、终点 B (0.30, 0.05, 0.48)、
  障碍球 (0.25, 0.05, 0.34) r = 0.07、避障途经点 W (0.42, -0.12, 0.36)
- 路径 A → W → B 两段线性插值（330 个轨迹点），全程与障碍保持 ≥ 0.10 m
- 相机：lookat (0.25, 0.05, 0.30)、distance 1.10、azimuth 60°、elevation -30°，
  黄色轨迹线用手算 lookAt 透视投影叠加（fovy = 45°）
