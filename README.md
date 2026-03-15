# PDKZero

PDKZero 是一个参考 [DouZero](https://github.com/kwai/DouZero) 思路实现的 4 人跑得快 AI 引擎。

当前版本包含：

- 4 人跑得快规则引擎
- 合法动作枚举与特殊规则处理
- 库优先的环境封装
- `random` / `heuristic` / `deep` 三类智能体接口
- PyTorch 候选动作打分模型
- 自博弈训练、对战和评估脚本

## 规则边界

规则来源于 [`rules.md`](./rules.md)，并已覆盖以下关键特殊项：

- 红桃 3 首出
- 炸弹压制
- `AA22` 双连对
- `3333` 只能出四带一
- 三带对子可压三带二
- 保本
- 放走包赔

## 环境准备

项目使用 `uv` 创建本地虚拟环境：

```bash
uv venv .venv
env UV_CACHE_DIR=$PWD/.uv-cache uv pip install --python .venv/bin/python pytest numpy torch --index-url https://download.pytorch.org/whl/cpu
```

## 运行测试

```bash
.venv/bin/python -m pytest -q
```

## 训练

```bash
.venv/bin/python train.py --device cpu --max-episodes 16 --batch-size 32 --num-eval-games 4
```

如果你有可用 CUDA，也可以把训练本身放到 GPU：

```bash
.venv/bin/python train.py --device cuda:0 --max-episodes 16 --batch-size 32 --num-eval-games 4
```

## 本地对战

```bash
.venv/bin/python play.py --games 1
```

## 浏览器人机对战

仓库默认已包含一个可运行的模型文件：

```text
checkpoints/model.pt
```

因此 `git clone` 后可以直接启动网页版。  
如果你想替换成自己训练的模型，再执行训练：

```bash
.venv/bin/python train.py --device cpu --max-episodes 200 --batch-size 64 --num-eval-games 4
```

然后启动网页版服务：

```bash
.venv/bin/python serve_web.py --host 0.0.0.0 --port 8000 --checkpoint checkpoints/model.pt
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

局域网其他设备访问：

```text
http://<你的机器局域网IP>:8000/
```

当前版本为：

- 单房间单局
- 人类固定座位 0
- 其他 3 家使用 `checkpoints/model.pt`
- 页面内点击合法动作按钮出牌

## 评估

```bash
.venv/bin/python evaluate.py --games 8
```

## 项目结构

```text
pdkzero/
  agents/      baseline and deep agents
  dmc/         candidate-scoring model and self-play training
  env/         observation encoding and gym-like wrapper
  game/        cards, move detection, legal action generation, engine
```
