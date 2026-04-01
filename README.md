# 模糊需求审查器

这是一个本地可运行的 Web 应用，用来验证这条链路：

1. 输入模糊需求
2. 系统把需求归一化并转成一组可检查规则
3. 对代码、文章或通用文本做规则审查
4. 输出综合评分、分项结果和修改建议

## 当前能力

- 后端：FastAPI
- 前端：独立 HTML 页面，位于 `frontend/index.html`
- 本地模式：启发式规则推断 + 简单打分
- 大模型模式：兼容 OpenAI 风格的 `chat/completions` 接口

## 启动方式

```bash
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

浏览器打开 `http://127.0.0.1:8000`

## 配置大模型 API

后端默认读取以下环境变量：

```bash
set OPENAI_API_KEY=你的密钥
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-4.1-mini
```

如果你接的是兼容 OpenAI 协议的其他平台，只需要把 `OPENAI_BASE_URL` 和 `OPENAI_MODEL` 改成对应值即可。

## API

### 本地规则审查

`POST /api/review`

```json
{
  "requirement": "这篇方案要更专业、结构清楚、覆盖完整，而且要给出可以执行的建议。",
  "artifact": "本方案将围绕用户增长展开说明。首先分析目标人群，其次给出渠道策略和执行节奏，最后总结关键指标。建议先从低成本渠道验证，再根据转化率调整资源投入。",
  "artifact_type": "article"
}
```

### 大模型审查

`POST /api/review/llm`

```json
{
  "requirement": "这篇方案要更专业、结构清楚、覆盖完整，而且要给出可以执行的建议。",
  "artifact": "本方案将围绕用户增长展开说明。首先分析目标人群，其次给出渠道策略和执行节奏，最后总结关键指标。建议先从低成本渠道验证，再根据转化率调整资源投入。",
  "artifact_type": "article",
  "model": "gpt-4.1-mini",
  "temperature": 0.2
}
```

### 查看大模型配置状态

`GET /api/llm/status`

## 建议的下一步演进

- 增加“规则确认”界面，让用户手工编辑系统生成的 rubric
- 针对代码接入 AST、lint、test、复杂度分析
- 针对文章接入事实核验、结构分析、受众匹配分析
- 增加审查历史、版本对比和人工复核流程
