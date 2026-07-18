# Raven Watch Agents Pro 中文研投终端

这是一个面向量化学习与研究的专业研投终端。它把 A 股、美股、港股和全球主要市场的行情、K 线与分时图、在线新闻与公告、基本面快照、Polymarket 预期证据、本地证据评分、多因子挖掘、多策略融合、12 角色智能体研判、风险控制和历史回测放在同一个工作台中，并同时提供 Web、Docker 与 Windows `.exe` 运行方式。

> 软件不自动下单，也不承诺收益。`20%` 是历史研究目标，必须同时查看时间顺序保留集、分段结果、成本压力和最大回撤；历史回测不代表未来表现。

## 主要功能

- 实时终端：K 线、MA5/MA20/MA60、成交量、分时价、均价线和 15 秒 SSE 自动刷新；界面同时显示交易所、资产类型、币种、行情时间、开闭市和延迟状态。
- 四市场接口：A 股、美股、港股与全球市场分别使用多级在线回退，单个提供方不可用时自动尝试下一来源。
- 全量证券库：随包提供 19,826 条可离线检索的目录快照，覆盖 5,536 只 A 股、11,464 只美股、2,809 只港股与 17 个全球精选标的；支持代码/名称、资产类型、板块分类、分页和来源状态筛选。
- 在线目录更新：A 股与港股使用新浪财经并保留东方财富回退，美股使用 Nasdaq Trader 官方目录；本地缓存每日后台刷新，断网或上游异常时自动使用最近缓存与随包快照。
- 全球证券搜索：按代码或名称联想 Microsoft Finance 在线目录，可筛选股票、ETF、指数、基金和 REIT，并显示交易所与币种；76 个常用标的与 35 个 ETF 仍固定置顶。
- 27 个可审计因子：覆盖趋势、动量、突破、反转、波动、回撤、量价、OBV 与相对强弱，并加入 12-1 月动量、波动调整动量、趋势效率、52 周高点距离、Amihud 流动性、Parkinson 区间波动、预期损失、Ulcer 指数、跳空风险、下行风险、MFI、ADX、隔夜和日内收益结构。
- 因子方法来源：新增因子返回公式、最低历史长度与论文/指标文档链接；历史不足时明确显示 `N/A`，不会用零分冒充有效信号。
- 5 类策略融合：趋势跟随、多周期动量、放量突破、均值回归、量价确认，并按市场状态动态调整权重。
- 滚动样本外验证：仅用过去 126/252 个交易日从三套固定画像中选择下一测试段策略，次日执行信号，并拼接成不回填训练收益的样本外净值。
- 量化稳健性：同时显示买入持有基准、成本压力、折次胜率、概率 Sharpe、最大回撤，以及 stationary bootstrap 风险区间；训练/测试窗口、手续费、滑点、仓位上限、压力倍数、风险期限、重采样次数、平均区块和随机种子均可量化调整。
- 因子挖掘：最多检验 20 个时间序列，显示 Rank IC、t 值、方向胜率、三段历史 IC 稳定度、样本数和最近 30 期七维因子热力图。
- 在线证据：聚合东方财富、Yahoo Finance、Nasdaq RSS 与 GDELT 原始媒体索引，并接入巨潮资讯、SEC EDGAR、HKEXnews 等官方披露；同一事件按标题、数值、类别与发布时间保守聚类，只有五个独立发布者共同报道才标记“五源验证”，官方原文单独保留且不会被媒体转载替代。
- 预测市场证据：通过 Polymarket Gamma 公共接口发现公司与宏观相关市场，按 Yes/No 概率、买卖价差、流动性、成交量、语义关联和事件影响方向进行质量校准；只有方向明确且达到门槛的市场才参与评分，其他结果只展示。快照原子写入本地缓存，断网和纯离线模式会明确标记缓存状态与年龄。
- 基本面快照：A 股、美股、港股和全球标的使用多提供方并发获取估值、成长、盈利、现金流和偿债指标；字段按来源可靠度、报告期新鲜度与跨源一致性选择，保留冲突值并显示质量分、报告期、字段来源和提供方状态。
- 双尺度研判：用户可见评分固定为 `0–100`，`50` 代表中性；独立保留 `-100–100` 多空方向与方向强度，偏空观点不再被误解为非法负分。
- 本地证据评分：技术、样本外量化、基本面、新闻和预测市场五类证据在本机确定性计算；预测市场基础权重限制为 `10%`，缺失证据不按中性分处理，而是按覆盖度与可靠度重新归一化权重，并依据来源质量、跨证据一致度和冲突向中性校准。界面逐步显示基础权重、可靠度、有效权重、方向贡献和最终校准。
- 综合风控：市场状态、策略共识、目标仓位、置信度、ATR 止损止盈与风险等级。
- 12 个投研角色：技术、情绪、新闻、基本面、多空研究、研究经理、交易员、三类风险分析师和组合经理。
- 离线规则模式：无需 API Key，可完整跑通研究流程；外部网络不可用时自动读取最近的 Polymarket 本地快照，没有缓存则保持未知并从评分中剔除。
- 在线 LLM 模式：DeepSeek V4 为默认在线服务，支持 `deepseek-v4-flash`、`deepseek-v4-pro`、思考模式、推理强度、连接诊断、限流重试和 Token 用量汇总；同时兼容 OpenAI、Qwen 与 Ollama。
- 智能体运行监控：异步任务、SSE 实时进度、逐角色耗时、取消、失败隔离和离线回退。
- 五类策略：杠杆战术增长、自适应验证、风险调整动量、经典动量轮动、双均线。
- 真实成本：手续费、滑点、下一交易日执行、仓位与风险预算。
- 风险指标：年化收益、波动率、夏普、索提诺、最大回撤、卡玛、VaR、CVaR、换手率和收益因子。
- 数据质量评分：按样本量、日期覆盖、缺失值、重复记录和异常 OHLCV 计算质量等级，在分析前直接提示历史数据是否充分。
- 训练/保留集验证：训练集选参，最后 30% 数据只做样本外验收。
- 固定规则验证：70/30 时间顺序切分、四个连续区段和手续费/滑点翻倍压力测试。
- 本地审计：保存行情、仓位、净值、参数排行榜、验证 JSON、图表和代理报告。

交互原型与设计基础保存在 [Raven Watch Agents Pro Figma 文件](https://www.figma.com/design/vglROiapGSJ05SIsjtEdVx)。

## 启动在线终端

```powershell
cd D:\codex\quant-starter
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe web_app.py --host 127.0.0.1 --port 8765
```

浏览器访问 `http://127.0.0.1:8765`，API 文档位于 `http://127.0.0.1:8765/docs`。

主要接口：

- `GET /api/dashboard`：历史 K 线、实时分时、因子、策略、风控与因子挖掘。
- `GET /api/markets`：四市场定义、预设数量和可用资产类型。
- `GET /api/symbols`：按市场和资产类型读取本地精选证券目录。
- `GET /api/instruments/catalog`：检索、筛选和分页读取四市场证券库，可用 `refresh=true` 触发在线更新。
- `GET /api/instruments/search`：按代码或名称搜索在线全球证券目录。
- `GET|POST /api/quant/walk-forward`：返回当前证券的滚动样本外净值、逐折结果、成本压力、稳健性评分和 bootstrap 风险区间；`POST` 可提交完整量化参数。
- `GET|POST /api/evidence`：并发获取新闻、基本面和样本外量化结果，并返回本地综合证据、覆盖度、冲突与逐来源状态。
- `GET|POST /api/polymarket`：返回相关预测市场、隐含概率、方向语义、质量、纳入理由和五步校准过程；`offline=true` 强制只读本地快照。
- `GET /api/market/live`：统一实时快照与分钟数据。
- `GET /api/stream`：SSE 实时行情流。
- `GET /api/news`：当前标的的多源在线新闻/公告、原文链接与逐来源状态。
- `GET /api/research/factors`：新增因子的公式、历史要求和研究来源目录。
- `GET /api/research/providers`：在线模型服务和服务端配置状态。
- `POST /api/research/providers/test`：使用当前模型、密钥和推理参数执行短连接诊断。
- `POST /api/research/jobs`：创建异步智能体研判任务。
- `GET /api/research/jobs/{job_id}/stream`：12 角色 SSE 运行进度。
- `GET /api/research/jobs/{job_id}`：查询状态和最终报告。
- `DELETE /api/research/jobs/{job_id}`：取消正在运行的任务。
- `POST /api/research/agents`：兼容同步调用的完整研判接口。

## 市场与在线来源

- A 股日线：腾讯直连、东方财富/AkShare、腾讯/AkShare；实时行情使用腾讯。
- 美股日线：Nasdaq 官方历史接口、Yahoo Chart、Microsoft Finance、yfinance、Stooq；实时行情依次尝试 Yahoo、Nasdaq 和 Microsoft Finance。
- 港股日线：腾讯直连、Microsoft Finance、东方财富/AkShare；实时行情依次尝试腾讯和 Microsoft Finance。
- 全球市场：Microsoft Finance 为主要日线与分时来源，并保留 Yahoo Chart 与 yfinance 回退。
- A 股基本面：Microsoft Finance 估值/市场数据、新浪财经财务指标与东方财富财务分析并发合并。
- 美股基本面：Microsoft Finance、Nasdaq 官方财务报表、SEC XBRL Company Facts、东方财富财务分析与 Yahoo/yfinance 多级回退。
- 港股基本面：Microsoft Finance、东方财富财务分析与 Yahoo/yfinance 并发合并；全球标的保留 Microsoft Finance 与 Yahoo/yfinance 回退。
- 证券目录：A 股和港股优先读取[新浪财经市场中心](https://vip.stock.finance.sina.com.cn/mkt/)，东方财富作为回退；美股读取 [Nasdaq Trader Symbol Directory](https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs)。目录在 `%LOCALAPPDATA%\RavenWatchAgentsPro\catalog` 中缓存，并随程序提供可离线检索的压缩快照。
- A 股新闻与披露：[巨潮资讯](https://www.cninfo.com.cn/)、[东方财富](https://finance.eastmoney.com/)、上交所/深交所披露入口与 [GDELT](https://www.gdeltproject.org/) 原始媒体索引。
- 美股新闻与披露：[SEC EDGAR](https://www.sec.gov/edgar/search/)、[Nasdaq RSS](https://www.nasdaq.com/nasdaq-rss-feeds)、[Yahoo Finance](https://finance.yahoo.com/)与 [GDELT](https://www.gdeltproject.org/)。
- 港股新闻与披露：[HKEXnews](https://www.hkexnews.hk/)、[Yahoo Finance](https://finance.yahoo.com/)与 [GDELT](https://www.gdeltproject.org/)。
- 预测市场：[Polymarket Gamma API](https://docs.polymarket.com/api-reference/introduction) 用于市场发现；概率数组映射与公共市场数据遵循[市场数据说明](https://docs.polymarket.com/market-data/overview)，紧价差时使用买卖中点、宽价差时优先最近成交价，规则依据[价格与订单簿说明](https://docs.polymarket.com/concepts/prices-orderbook)。缓存位于 `%LOCALAPPDATA%\RavenWatchAgentsPro\polymarket`。

预测市场价格是参与者对事件结果的交易定价，不是新闻事实、公司基本面或必然结果。事件对股票方向的映射采用保守词典和明确门槛；语义不确定、非二元、已结束、低关联或低质量市场不会进入综合分。在线提供方不可用时，界面会显示 `在线源不可用`、`断网回退缓存` 或 `纯离线缓存`，不会关闭 TLS 校验或静默填入模拟概率。

终端每 15 秒重新获取一次可用行情，但这不等于所有交易所都提供零延迟数据。实际时效取决于交易所规则和上游授权；界面会显示行情时间、开闭市状态和估算延迟，研究时应以这些字段为准。

## 滚动样本外量化验证

Web 工作台的“策略实验室”不会把训练期收益计入展示结果。系统在每个滚动训练窗口中只比较三套预先固定的策略画像，并将选中的画像用于紧随其后的测试窗口；目标仓位统一延迟一个交易日执行。默认成本为手续费万分之三加滑点万分之二，同时提供双倍成本压力结果。

点击策略实验室右上角的参数按钮可以调整训练/测试窗口、仓位上限、手续费与滑点（基点）、压力倍数、bootstrap 风险期限、模拟次数、平均区块长度和随机种子。`自动`窗口会依据当前历史长度选择可执行配置；固定随机种子使相同数据与参数可复现。提高模拟次数只能降低重采样噪声，不能消除数据偏差、模型选择偏差或未来制度变化。

稳健性评分综合逐折超额胜率、样本外概率 Sharpe、相对基准、最大回撤、成本敏感度和有效样本长度。概率 Sharpe 的有限样本修正参考 [Bailey 与 López de Prado 的研究](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)，相关性风险区间使用 [Politis 与 Romano 的 stationary bootstrap](https://doi.org/10.1080/01621459.1994.10476870)。这些指标用于暴露不稳定性，不构成未来收益预测。

长期使用 SEC 数据时，建议按 SEC 访问规范设置可联系的应用标识：

```powershell
$env:RAVENWATCHAGENTSPRO_SEC_USER_AGENT = "YourApp/1.0 your-email@example.com"
```

## 在线智能体

点击终端右上角“开始研判”，选择“在线模型”。DeepSeek V4 已作为默认服务，默认模型为速度优先的 `deepseek-v4-flash`，也可切换为 `deepseek-v4-pro`；填写 API Key 后可先点击“测试连接”。默认编排包含 12 个角色步骤，可调整思考模式、推理强度、四类分析师、多空辩论轮数、风险合议轮数，以及是否补充新闻和基本面数据。

支持的默认兼容地址：

- OpenAI：`https://api.openai.com/v1`
- DeepSeek V4：`https://api.deepseek.com`
- Qwen：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- Ollama：`http://127.0.0.1:11434/v1`

DeepSeek 当前推荐模型 ID 为 `deepseek-v4-flash` 与 `deepseek-v4-pro`，接口与参数以 [DeepSeek API 文档](https://api-docs.deepseek.com/api/create-chat-completion)为准。程序只使用响应中的最终 `content` 生成报告，不展示或保存模型的内部推理内容；遇到限流或临时服务错误会进行有上限的短暂重试。

也可以只在服务端配置模型，不在每次研判时输入密钥。DeepSeek 专用配置示例：

```powershell
$env:RAVENWATCHAGENTS_DEEPSEEK_MODEL = "deepseek-v4-flash"
$env:RAVENWATCHAGENTS_DEEPSEEK_API_KEY = "填写DeepSeekAPIKey"
$env:RAVENWATCHAGENTS_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

通用环境变量如下：

```powershell
$env:RAVENWATCHAGENTS_LLM_PROVIDER = "openai"
$env:RAVENWATCHAGENTS_LLM_MODEL = "填写模型ID"
$env:RAVENWATCHAGENTS_LLM_API_KEY = "填写APIKey"
# 可选：覆盖 OpenAI 兼容地址
$env:RAVENWATCHAGENTS_LLM_BASE_URL = "https://api.openai.com/v1"
```

每个服务也可单独配置，例如 `RAVENWATCHAGENTS_OPENAI_MODEL`、`RAVENWATCHAGENTS_OPENAI_API_KEY`、`RAVENWATCHAGENTS_OPENAI_BASE_URL`；DeepSeek、Qwen、Ollama 分别替换变量名中的 `OPENAI`。程序也会读取 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY` 和 `DASHSCOPE_API_KEY`。

前端输入的 API Key 只随本次任务发给本机 API，启动成功后立即清空，不写入 `localStorage`、报告或日志。生产部署应通过 HTTPS 和服务端环境变量提供密钥。

## Docker 部署

```powershell
docker build -t raven-watch-agents-pro .
docker run --rm -p 8765:8765 raven-watch-agents-pro
```

部署到公网时，应在容器前配置 HTTPS、身份认证、访问频率限制和反向代理；默认 API 适合本机或受信网络研究，不应直接裸露到公网。

## 启动新版桌面终端

源码运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe web_desktop_app.py
```

打包后的现代桌面程序：

```text
dist\RavenWatchAgentsPro\RavenWatchAgentsPro.exe
```

桌面端仅监听随机的 `127.0.0.1` 端口，并用一次性会话令牌保护本地 API。运行日志位于 `%LOCALAPPDATA%\RavenWatchAgentsPro\logs\desktop.log`。

## 启动经典桌面工作台

```powershell
cd D:\codex\quant-starter
.\.venv\Scripts\python.exe raven_watch_agents_app.py
```

默认输出位置：

- 投研报告：`Documents\RavenWatchAgentsCN\research`
- 回测结果：`Documents\QuantStarter\outputs`
- 决策记忆：`Documents\RavenWatchAgentsCN\memory\decision_log.jsonl`

API Key 仅保存在当前软件进程内，不写入报告、日志或配置文件。

在投研页选择市场后，可点击代码框旁的证券库按钮，按代码、名称、资产类型或板块筛选全部目录；也可从精选标的中选择或直接输入交易所代码。点击“查看K线”即可先检查行情，行情标题会同步显示样本量与质量等级。`Ctrl+K` 可快速加载当前个股K线。完成投研后，报告目录会额外保存 `kline_chart.png`。

回测页提供同一套个股选择器和“个股K线”标签。双均线回测会自动载入所选个股，并在启用“保存图表”时输出 `kline_chart.png`。

## 自适应策略

桌面回测页默认使用 `adaptive`：

1. 在前 70% 数据上比较 48 组风险调整动量参数。
2. 评分同时考虑收益、夏普、卡玛、最大回撤和换手率。
3. 选中的参数应用到最后 30% 保留集。
4. 分别显示训练集与保留集是否达到目标年化和回撤约束。

该流程会生成：

- `parameter_leaderboard.csv`
- `holdout_validation.json`
- `metrics.csv`
- `target_weights.csv`
- `equity_curve.csv`
- `equity_vs_benchmark.png`
- `drawdown.png`
- `allocation.png`

## 杠杆战术增长

桌面回测页默认选择“杠杆战术增长（高风险）”，使用 `QQQ` 生成信号，在 `TQQQ` 与短期国债 ETF `BIL` 之间配置：

1. 统计 QQQ 21、63、126、252 日收益中为正的数量，至少两项为正。
2. QQQ 必须位于 200 日均线上方，且 21 日年化波动率低于 30%。
3. TQQQ 仓位按 63 日波动率缩放，目标波动 55%，最高仓位 100%。
4. 风险条件不满足或剩余仓位配置到 BIL。
5. 当日收盘生成目标仓位，下一交易日执行，并计入手续费与滑点。

截至 2026-07-13 的 Yahoo Chart 复权日线示例中，单边成本 0.05% 时：完整回测年化约 30.6%，正式验证段年化约 31.6%，后 30% 保留段年化约 25.3%，四个连续区段年化均超过 20%；成本翻倍后的保留段年化约 24.9%。同一结果的历史最大回撤约为 49.5%，年化换手超过 1000%，属于高风险研究策略，不适合作为收益承诺。

该模式生成：

- `tactical_growth_diagnostics.csv`
- `tactical_growth_validation.json`
- `trades.csv`
- `prices.csv`
- `target_weights.csv`
- `metrics.csv`
- 净值、回撤和仓位图表

## 命令行回测

默认模拟数据与自适应验证：

```powershell
.\.venv\Scripts\python.exe run_demo.py
```

A 股：

```powershell
.\.venv\Scripts\python.exe run_demo.py --source a-share --symbols 600519 000001 300750 --start 2020-01-01 --end 2026-07-10
```

纳斯达克/美股：

```powershell
.\.venv\Scripts\python.exe run_demo.py --source nasdaq --symbols AAPL MSFT NVDA AMZN GOOGL --start 2020-01-01 --end 2026-07-10
```

杠杆战术增长与时间顺序验证：

```powershell
.\.venv\Scripts\python.exe run_demo.py --strategy tactical-growth --source nasdaq --start 2010-03-01
```

固定风险调整动量：

```powershell
.\.venv\Scripts\python.exe run_demo.py --strategy risk-momentum --lookback 126 --skip-recent 21 --trend-window 100 --target-volatility 0.18 --max-position 0.65
```

经典双均线：

```powershell
.\.venv\Scripts\python.exe run_demo.py --strategy ma --ticker ALPHA --fast 20 --slow 120
```

本地 CSV：

```powershell
.\.venv\Scripts\python.exe run_demo.py --source csv --prices-csv path\to\prices.csv
```

价格表 CSV 格式：

```csv
date,AAA,BBB,CCC
2024-01-02,100.0,50.0,80.0
2024-01-03,101.2,49.8,81.0
```

## 构建 EXE

构建现代研投终端：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Build-Raven-Exe.ps1
```

构建产物：

```text
dist\RavenWatchAgentsPro\RavenWatchAgentsPro.exe
```

构建经典 Tkinter 工作台：

```powershell
.\Build-Exe.ps1
```

构建产物：

```text
dist\RavenWatchAgentsCN\RavenWatchAgentsCN.exe
```

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe raven_watch_agents_app.py --smoke-test
.\.venv\Scripts\python.exe raven_watch_agents_app.py --data-smoke-test
.\.venv\Scripts\python.exe quant_app.py --smoke-test
.\.venv\Scripts\python.exe web_app.py --smoke-test
.\.venv\Scripts\python.exe web_desktop_app.py --smoke-test
.\.venv\Scripts\python.exe web_desktop_app.py --window-smoke-test
```

因子挖掘中的 IC 是单标的时间序列 Rank IC，用于检查该标的历史上的预测一致性；它不等同于跨股票截面的行业中性因子 IC，也不构成未来收益保证。

## 代码结构

模块依赖、评分语义与扩展约定见 [ARCHITECTURE.md](ARCHITECTURE.md)。

```text
raven_watch_agents_app.py          经典 Tkinter 桌面研究与回测入口
web_desktop_app.py                 WebView2 桌面入口与本地 API 隔离
web_app.py                         FastAPI 在线服务、缓存与 SSE
web/                               专业研投终端前端与本地化图表依赖
quant_app.py                       内嵌回测工作台
run_demo.py                        命令行入口
src/quant_starter/data.py          多源行情与回退
src/quant_starter/global_market.py Microsoft Finance 全球目录、报价与 OHLCV
src/quant_starter/realtime.py      四市场实时快照、分钟行情与延迟状态
src/quant_starter/factors.py       27 因子、市场状态、5 策略融合与 20 序列 IC 稳定性挖掘
src/quant_starter/news.py          多市场新闻、官方公告、事件聚类与五源交叉验证
src/quant_starter/research_data.py 多市场基本面、字段级证据与质量评分
src/quant_starter/local_evidence.py 本地新闻、基本面与综合证据评分
src/quant_starter/scoring.py       方向、评级、阈值与可靠度校准契约
src/quant_starter/numeric.py       有限数值、边界与秩相关公共工具
src/quant_starter/instrument_catalog.py 四市场证券目录、在线刷新与离线缓存
src/quant_starter/walk_forward.py 滚动样本外选择、成本压力与 bootstrap 风险
src/quant_starter/strategies.py    均线、动量、风险调整动量
src/quant_starter/optimization.py  训练/保留集参数验证
src/quant_starter/validation.py    固定规则、分段与成本压力验证
src/quant_starter/backtest.py      下一日执行与交易成本
src/quant_starter/metrics.py       收益和风险指标
src/quant_starter/symbols.py       个股预设、显示与代码校验
src/quant_starter/kline.py         蜡烛图、均线与成交量图
src/quant_starter/agent_workflow.py 多智能体编排
src/quant_starter/llm_client.py     DeepSeek V4 与 OpenAI 兼容客户端、重试和用量统计
src/quant_starter/research_store.py 报告与记忆落盘
```
