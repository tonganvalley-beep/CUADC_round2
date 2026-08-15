# 天井关键点边缘位置与透视裁剪分析

`analyze_well_position.py` 使用 `step1_0724.pt` 对 `unnormal_pic` 的整帧图进行检测，并严格复用
`detect_well_exposure(4).py` 的第一阶段参数和透视裁剪规则：

- 先将整帧缩放至 960×540，按 `imgsz=640, conf=0.60, iou=0.45` 推理，再映射回原图坐标；
- 关键点 1～4 的置信度均须不低于 0.35；
- 检查四边形凸性、面积、边长和对边平行度；
- 四角向外扩 3 px，透视变换为 100×100。

## 运行

本机可用的 YOLO 环境为：

```powershell
& 'D:\Anaconda\envs\yolo\python.exe' analyze_well_position.py
```

界面中，左侧是整帧的检测框、四角框和关键点位置，右侧是当前天井的 100×100 透视裁剪图。

- `1`：透视正常（NORMAL）
- `0`：透视异常（ABNORMAL）
- `U`：无法确定（UNCERTAIN）
- `A / D`：上一项 / 下一项
- `Z`：撤销
- `Q / Esc`：保存并退出

正式人工标签必须由用户在界面中逐项提交。`NORMAL` 表示井牌主体完整、四边被合理拉正且没有明显透视变形；`ABNORMAL` 表示原图边界截断了
井牌/关键点，或裁剪结果明显缺失、错误变形；因过暗或过曝而无法可靠判断几何形状时记为 `UNCERTAIN`，不强行归类。

只生成分析结果、不打开标注窗口：

```powershell
& 'D:\Anaconda\envs\yolo\python.exe' analyze_well_position.py --no-gui
```

## 输出

默认输出在 `well_position_analysis/`：

- `annotated_frames/`：整帧关键点、外围框、最近边缘点及测距线；
- `crops/`：与线上逻辑一致的 100×100 透视裁剪；
- `review_panels/`：整帧与当前裁剪并排的复核图；
- `contact_sheets/`：每页 4 个天井的快速总览图；
- `well_detections.csv`：每个天井的框、5 个关键点、裁剪状态、透视标签和边缘距离；
- `frame_summary.csv`：每张整帧中最靠近边缘的关键点及像素/相对距离；
- `perspective_labels.csv`：可续标的人工透视标签；
- `detections.json`：完整原始分析数据。

`edge_distance_px` 是关键点到左、右、上、下四条图像边界的最小像素距离。
`edge_distance_rel` 使用对应方向的图像尺寸归一化，例如靠左/右时除以 `width-1`，靠上/下时除以
`height-1`。`edge_margin_px/rel` 是带符号的边界余量：负值表示模型预测点已落在画面外。帧级“最边缘点”
按最小带符号余量选择，这样越界点会优先暴露；`edge_distance` 仍保持非负，表示到对应边界的实际距离。
因此后续既可比较绝对像素阈值，也可比较不同分辨率下的相对位置阈值。
