from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

ArtifactType = Literal["auto", "code", "article", "general"]


@dataclass
class Rule:
    id: str
    name: str
    description: str
    weight: float
    check_points: list[str]
    evidence_keywords: list[str]


@dataclass
class RuleResult:
    rule_id: str
    rule_name: str
    score: float
    passed: bool
    reason: str
    evidence: list[str]


class ReviewRequest(BaseModel):
    requirement: str = Field(..., min_length=5, description="模糊需求描述")
    artifact: str = Field(..., min_length=5, description="待审查内容")
    artifact_type: ArtifactType = "auto"


class LLMReviewRequest(ReviewRequest):
    model: str | None = Field(default=None, description="覆盖默认模型名")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class ReviewResponse(BaseModel):
    inferred_artifact_type: str
    normalized_requirement: str
    rules: list[dict]
    overall_score: float
    passed: bool
    summary: str
    suggestions: list[str]
    results: list[dict]


class LLMConfigResponse(BaseModel):
    configured: bool
    base_url: str
    model: str
    api_key_present: bool


app = FastAPI(
    title="模糊需求审查器",
    description="把模糊需求转换为可执行规则，并审查生成物是否达标。",
    version="0.2.0",
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


KEYWORD_GROUPS = {
    "clarity": ["清晰", "易懂", "明确", "结构", "层次", "可读", "readable", "clear"],
    "completeness": ["完整", "全面", "覆盖", "细节", "场景", "边界", "complete", "coverage"],
    "accuracy": ["准确", "正确", "严谨", "可靠", "无误", "accurate", "correct"],
    "actionable": ["可执行", "落地", "步骤", "方案", "建议", "actionable", "practical"],
    "safety": ["安全", "风险", "合规", "限制", "禁止", "safe", "policy"],
    "style": ["专业", "简洁", "有说服力", "正式", "友好", "风格", "tone", "style"],
}

STOPWORDS = {
    "一个",
    "这个",
    "那个",
    "我们",
    "你要",
    "可以",
    "根据",
    "需要",
    "要求",
    "进行",
    "是否",
    "达到",
    "规则",
    "内容",
    "生成",
    "审查",
    "应用",
}


def infer_artifact_type(text: str, requested_type: ArtifactType) -> str:
    if requested_type != "auto":
        return requested_type

    code_markers = [
        "def ",
        "class ",
        "function ",
        "const ",
        "let ",
        "import ",
        "from ",
        "{",
        "}",
        "```",
        "return ",
        "if (",
    ]
    article_markers = ["标题", "引言", "总结", "观点", "段落", "因此", "首先", "其次"]

    code_hits = sum(marker in text for marker in code_markers)
    article_hits = sum(marker in text for marker in article_markers)

    if code_hits >= 3:
        return "code"
    if article_hits >= 2 or len(re.findall(r"[。！？]", text)) >= 4:
        return "article"
    return "general"


def normalize_requirement(requirement: str) -> str:
    cleaned = re.sub(r"\s+", " ", requirement).strip()
    cleaned = re.sub(r"(要求求|要我想做一个应用，)", "", cleaned)
    return cleaned


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fffA-Za-z_]{2,}", text.lower())


def top_terms(text: str, limit: int = 8) -> list[str]:
    counts = Counter(tokenize(text))
    ranked = [term for term, _ in counts.most_common() if term not in STOPWORDS]
    return ranked[:limit]


def pick_focus_dimensions(requirement: str) -> list[str]:
    focus = []
    lowered = requirement.lower()
    for key, words in KEYWORD_GROUPS.items():
        if any(word.lower() in lowered for word in words):
            focus.append(key)
    if not focus:
        focus = ["clarity", "completeness", "accuracy", "actionable"]
    return focus


def build_rules(requirement: str, artifact_type: str) -> list[Rule]:
    focus = pick_focus_dimensions(requirement)
    terms = top_terms(requirement)
    rules: list[Rule] = []

    descriptions = {
        "clarity": "输出应当结构清晰、表达明确，读者可以快速理解核心内容。",
        "completeness": "输出应覆盖主要需求、关键场景和必要细节，不应遗漏核心点。",
        "accuracy": "输出应避免明显错误、冲突或与需求不一致的内容。",
        "actionable": "输出应包含可执行建议、步骤、判断标准或下一步行动。",
        "safety": "输出应考虑风险、限制、合规性或潜在误用问题。",
        "style": "输出的语气、形式和专业度应符合需求场景。",
    }
    checkpoints = {
        "clarity": ["是否有清晰结构", "是否存在模糊跳跃表达", "是否易于快速阅读"],
        "completeness": ["是否覆盖主要目标", "是否包含关键边界条件", "是否遗漏必要信息"],
        "accuracy": ["是否与需求相符", "是否存在明显自相矛盾", "是否有不可靠断言"],
        "actionable": ["是否给出步骤或建议", "是否便于执行", "是否可用于决策"],
        "safety": ["是否提示风险", "是否体现限制条件", "是否避免明显违规方向"],
        "style": ["是否符合预期风格", "是否保持一致语气", "是否专业自然"],
    }

    base_weight = round(1 / max(len(focus) + 1, 1), 2)
    for index, dim in enumerate(focus, start=1):
        rules.append(
            Rule(
                id=f"R{index}",
                name=dim,
                description=descriptions[dim],
                weight=base_weight,
                check_points=checkpoints[dim],
                evidence_keywords=terms[:4],
            )
        )

    if artifact_type == "code":
        rules.append(
            Rule(
                id=f"R{len(rules) + 1}",
                name="code_quality",
                description="代码应具有基本可读性、合理结构，并体现必要注释或命名语义。",
                weight=0.2,
                check_points=["是否有函数或模块结构", "命名是否具备语义", "是否存在基础说明"],
                evidence_keywords=["def", "class", "return", "import"],
            )
        )
    elif artifact_type == "article":
        rules.append(
            Rule(
                id=f"R{len(rules) + 1}",
                name="article_flow",
                description="文章应具备引入、展开和结论等基本组织方式。",
                weight=0.2,
                check_points=["是否有开头铺垫", "是否有主体展开", "是否有总结或结论"],
                evidence_keywords=["首先", "其次", "最后", "总结"],
            )
        )
    else:
        rules.append(
            Rule(
                id=f"R{len(rules) + 1}",
                name="fit_for_purpose",
                description="内容应直接服务于需求目标，而不是只做泛泛描述。",
                weight=0.2,
                check_points=["是否贴合目标", "是否避免空泛", "是否具备可判断性"],
                evidence_keywords=terms[:3],
            )
        )

    total_weight = sum(rule.weight for rule in rules) or 1
    for rule in rules:
        rule.weight = round(rule.weight / total_weight, 2)
    return rules


def keyword_evidence(text: str, keywords: Iterable[str]) -> list[str]:
    lowered = text.lower()
    evidence = [keyword for keyword in keywords if keyword and keyword.lower() in lowered]
    return evidence[:6]


def score_rule(rule: Rule, artifact: str, requirement: str, artifact_type: str) -> RuleResult:
    lowered_artifact = artifact.lower()
    evidence = keyword_evidence(lowered_artifact, rule.evidence_keywords + top_terms(requirement, 6))

    score = 45.0 + min(len(evidence) * 8, 24)
    reasons: list[str] = []

    lines = [line for line in artifact.splitlines() if line.strip()]
    if len(lines) >= 3:
        score += 6
        reasons.append("内容具备一定展开。")

    if any(token in lowered_artifact for token in ["因此", "所以", "because", "step", "步骤", "建议"]):
        score += 6
        reasons.append("出现了推理或行动性表达。")

    if rule.name in {"clarity", "article_flow"} and re.search(r"(1\.|2\.|一、|二、|首先|其次|最后)", artifact):
        score += 8
        reasons.append("结构化表达较明显。")

    if artifact_type == "code":
        if re.search(r"\b(def|class|function|const|let|return|if)\b", lowered_artifact):
            score += 10
            reasons.append("具备代码结构信号。")
        if "#" in artifact or "//" in artifact:
            score += 4
            reasons.append("包含注释或解释。")

    if artifact_type == "article":
        if len(re.findall(r"[。！？]", artifact)) >= 4:
            score += 8
            reasons.append("文章连贯度较好。")
        if "总结" in artifact or "结论" in artifact:
            score += 4
            reasons.append("存在收束段落。")

    if rule.name == "safety" and not re.search(r"(风险|限制|注意|合规|禁止|不要)", artifact):
        score -= 14
        reasons.append("风险或限制表达偏少。")

    if rule.name == "accuracy" and len(evidence) < 2:
        score -= 8
        reasons.append("与需求关键词的对齐证据偏弱。")

    if rule.name == "actionable" and not re.search(r"(步骤|建议|应该|可以|先|然后)", artifact):
        score -= 10
        reasons.append("可执行动作描述不足。")

    bounded = max(0.0, min(score, 100.0))
    if not reasons:
        reasons.append("规则命中证据一般，建议补充更明确内容。")

    return RuleResult(
        rule_id=rule.id,
        rule_name=rule.name,
        score=round(bounded, 1),
        passed=bounded >= 65,
        reason=" ".join(reasons),
        evidence=evidence,
    )


def summarize(artifact_type: str, overall_score: float, failed_rules: list[RuleResult]) -> str:
    if not failed_rules:
        return f"该{artifact_type}生成物整体上已经较好贴合需求，可视为基本达到预期。"

    weakest = ", ".join(result.rule_name for result in failed_rules[:3])
    return (
        f"系统已将模糊需求转为可检查规则。当前{artifact_type}生成物的综合得分为 {overall_score}，"
        f"薄弱点主要集中在 {weakest}。"
    )


def build_suggestions(results: list[RuleResult], artifact_type: str) -> list[str]:
    suggestions: list[str] = []
    for result in results:
        if result.passed:
            continue
        if result.rule_name == "clarity":
            suggestions.append("增加标题、分点或步骤编号，让结构更清楚。")
        elif result.rule_name == "completeness":
            suggestions.append("补充遗漏的关键场景、边界条件或输出示例。")
        elif result.rule_name == "accuracy":
            suggestions.append("让内容与需求关键词逐项对齐，删除无法确认的断言。")
        elif result.rule_name == "actionable":
            suggestions.append("加入更明确的下一步动作、执行步骤或判断标准。")
        elif result.rule_name == "safety":
            suggestions.append("补充风险、限制条件和不建议的做法。")
        elif result.rule_name == "style":
            suggestions.append("统一语气与表达风格，让内容更符合目标场景。")
        elif result.rule_name == "code_quality":
            suggestions.append("优化代码命名、结构和注释，让实现更容易维护。")
        elif result.rule_name == "article_flow":
            suggestions.append("补出引言、主体和结论三段式结构。")
        else:
            suggestions.append("让内容更直接回应需求目标，减少空泛表述。")

    if artifact_type == "code":
        suggestions.append("后续可以接入 AST、单测和静态分析，把规则审查升级为更强的代码审核。")
    return suggestions[:6]


def run_local_review(payload: ReviewRequest) -> ReviewResponse:
    normalized_requirement = normalize_requirement(payload.requirement)
    artifact_type = infer_artifact_type(payload.artifact, payload.artifact_type)
    rules = build_rules(normalized_requirement, artifact_type)
    results = [score_rule(rule, payload.artifact, normalized_requirement, artifact_type) for rule in rules]
    overall_score = round(sum(result.score * rule.weight for result, rule in zip(results, rules)), 1)
    failed = [result for result in sorted(results, key=lambda item: item.score) if not result.passed]
    passed = overall_score >= 70 and all(result.score >= 55 for result in results)

    return ReviewResponse(
        inferred_artifact_type=artifact_type,
        normalized_requirement=normalized_requirement,
        rules=[asdict(rule) for rule in rules],
        overall_score=overall_score,
        passed=passed,
        summary=summarize(artifact_type, overall_score, failed),
        suggestions=build_suggestions(sorted(results, key=lambda item: item.score), artifact_type),
        results=[asdict(result) for result in results],
    )


def get_llm_settings(model_override: str | None = None) -> tuple[str, str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = (model_override or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")).strip()
    return api_key, base_url, model


def extract_json_block(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError("模型返回中未找到 JSON 对象。")
    return json.loads(match.group(0))


async def run_llm_review(payload: LLMReviewRequest) -> ReviewResponse:
    api_key, base_url, model = get_llm_settings(payload.model)
    if not api_key:
        raise HTTPException(status_code=400, detail="未配置 OPENAI_API_KEY，无法调用大模型审查接口。")

    normalized_requirement = normalize_requirement(payload.requirement)
    artifact_type = infer_artifact_type(payload.artifact, payload.artifact_type)
    fallback_review = run_local_review(payload)

    system_prompt = """
你是一个审查系统。你的工作是把模糊需求转成规则，然后严格评审输入内容。
你必须只返回 JSON，不要输出 Markdown，不要输出解释文字。
返回字段必须包含：
inferred_artifact_type, normalized_requirement, rules, overall_score, passed, summary, suggestions, results

要求：
1. rules 是对象数组，每个对象包含 id, name, description, weight, check_points, evidence_keywords
2. results 是对象数组，每个对象包含 rule_id, rule_name, score, passed, reason, evidence
3. overall_score 为 0 到 100 的数字
4. passed 为布尔值
5. suggestions 为字符串数组
6. 所有内容使用中文
7. 规则数量控制在 4 到 7 条
8. 输出必须是合法 JSON
""".strip()

    user_prompt = {
        "requirement": normalized_requirement,
        "artifact_type": artifact_type,
        "artifact": payload.artifact,
        "reference_rules": fallback_review.rules,
        "reference_summary": fallback_review.summary,
    }

    request_payload = {
        "model": model,
        "temperature": payload.temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_payload,
        )

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    body = response.json()
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if part.get("type") == "text")

    try:
        parsed = extract_json_block(content)
        return ReviewResponse.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"大模型返回格式无法解析: {exc}") from exc


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="前端页面不存在。")
    return FileResponse(index_file)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/llm/status", response_model=LLMConfigResponse)
def llm_status() -> LLMConfigResponse:
    api_key, base_url, model = get_llm_settings()
    return LLMConfigResponse(
        configured=bool(api_key),
        base_url=base_url,
        model=model,
        api_key_present=bool(api_key),
    )


@app.post("/api/review", response_model=ReviewResponse)
def review(payload: ReviewRequest) -> ReviewResponse:
    return run_local_review(payload)


@app.post("/api/review/llm", response_model=ReviewResponse)
async def review_with_llm(payload: LLMReviewRequest) -> ReviewResponse:
    return await run_llm_review(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
