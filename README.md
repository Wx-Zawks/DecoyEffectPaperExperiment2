# DIM + PRM 雾计算任务卸载论文复现

本项目提供基于论文《Research On Incentive Mechanisms for task offloading in fog computing based on Decoy Effect》的 Python 复现代码，实现了：

- `DIM` 诱饵激励机制
- `PRM` 偏好逆转推送机制
- 论文第 4.1 / 4.2 节的核心仿真实验
- 个体理性、真实性、计算高效性验证
- `JSON + CSV + matplotlib PNG + HTML` 结果导出

## 运行方式

先安装依赖：

```bash
pip install -r requirements.txt
```

直接运行完整论文复现：

```bash
python run_experiment.py
```

使用自定义配置覆盖默认参数：

```bash
python run_experiment.py experiments/default_python.json
```

在已有 `paper_results.json` 基础上重新生成图表：

```bash
python scripts/plot_results.py
python scripts/plot_results.py outputs_py/paper_results.json
```

## 默认实验参数

- 雾节点数 `I = 10, 20, 30, 40, 50, 60`
- 任务数 `J = 50`
- 雾节点 / 用户终端时钟频率 `f_i, f_j ∈ [1, 1.5] GHz`
- 任务报酬 `v_j ∈ [2, 20]`
- 任务时间成本 `t_j ∈ [200, 2000]`
- 价值函数参数 `α = β = 0.88`
- 损失厌恶 `λ = 2.25`
- 偏好系数 `δ_i` 为 `0~1` 截断正态分布，均值 `d` 可配置，默认 `0.5`

## 主要代码结构

- `repro/models.py`：雾节点、任务、平台回合结果等实体
- `repro/formulas.py`：论文公式 `(2)(3)(4)(5)(15)(16)(17)(20)(21)(24)(26)` 的实现
- `repro/dim.py`：算法 1 / 算法 2、诱饵参数与 `K` 强度系数
- `repro/prm.py`：定理 4-1 / 4-2、分组与推送诱饵设计
- `repro/platform.py`：平台调度、Vickrey 次低价拍卖、真实性监管、整轮仿真
- `repro/experiment.py`：全部实验批量运行、CSV/JSON 写出、PNG 图表生成
- `repro/svg.py`：基于 `matplotlib` 的论文风格绘图

## 输出目录

默认输出到 `outputs_py/`：

- `paper_results.json`：完整实验结果
- `summary.json`：汇总摘要
- `summary.csv`：关键指标汇总表
- `csv/`：分实验 CSV
- `figures/`：全部 PNG 图和 `index.html`

## 复现实验覆盖

- 诱饵参数 `χ / γ` 对 `K` 与目标任务被选率的影响
- 加入诱饵前后任务选择数量对比
- 高时间成本任务卸载分布比较
- `F / R / RF` 三种诱饵策略比较
- `PRM / DIM` 在参与节点数、卸载任务数、任务级出价、成交报酬、用户总效用上的对比
- 偏好系数均值 `d` 对 `DIM / PRM` 参与节点数的影响
- 个体理性、真实性、计算高效性验证
