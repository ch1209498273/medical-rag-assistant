"""Executable contract for the public portfolio documentation."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.public_release_manifest import PUBLIC_RELEASE_MANIFEST

PROJECT_ROOT = Path(__file__).resolve().parents[3]
README = PROJECT_ROOT / "README.md"
PUBLIC_DOCS = (
    README,
    PROJECT_ROOT / "SECURITY.md",
    PROJECT_ROOT / "ROADMAP.md",
    *(PROJECT_ROOT / "docs" / name for name in (
        "architecture.md",
        "evaluation.md",
        "engineering-decisions.md",
        "multi-agent-development.md",
        "portfolio-brief.md",
        "project-case-study.md",
        "demo-guide.md",
        "release-evidence.md",
    )),
)
PROJECT_DOCS = (
    PROJECT_ROOT / "docs" / "portfolio-brief.md",
    PROJECT_ROOT / "docs" / "project-case-study.md",
    PROJECT_ROOT / "docs" / "evaluation.md",
)

EXPECTED_README_HEADINGS = (
    "# 医疗知识问答助手",
    "## 30秒了解项目",
    "## 界面预览",
    "## 快速体验",
    "## 已实现功能",
    "## 系统架构",
    "## 一个问题的数据旅程",
    "## 评测与回滚",
    "## 工程质量",
    "## 多 Agent 开发",
    "## 安全与隐私",
    "## 技术决策",
    "## 当前限制",
    "## v2.0 路线",
    "## License",
)

_MARKDOWN_LINK_RE = re.compile(r"!?(?:\[[^]]+\])\(([^)\s]+)")
_CLAUSE_BOUNDARY_RE = re.compile(r"[\n。！？.!?;；,，、:：]")
_NEGATION_RE = re.compile(
    r"(?:不|非|未|无|没有|不能|不可|尚未|尚无|不等于|不构成|"
    r"not|no|never|cannot|can't|doesn't|does not|without|not yet)",
    re.IGNORECASE,
)
_FUTURE_RE = re.compile(
    r"(?:候选|路线|未来|计划|考虑|留到|v2\.?0|roadmap|future|planned|candidate)",
    re.IGNORECASE,
)

_PRODUCTION_CLAIM_PATTERNS = (
    re.compile(
        r"(?i)\b(?:production[- ]ready|medical[- ]grade|"
        r"production[- ]grade|"
        r"clinical(?:ly)?[- ](?:ready|validated|safe)|"
        r"safe[- ]for[- ]clinical[- ]use)\b"
    ),
    re.compile(
        r"(?i)\b(?:ready|validated|approved|certified|safe|suitable|usable)\b"
        r"[^\n.!?。！？,，、:：;；]{0,32}\b(?:production|medical|clinical)\b"
    ),
    re.compile(
        r"(?i)\b(?:production|medical|clinical)\b"
        r"[^\n.!?。！？,，、:：;；]{0,32}\b(?:ready|validated|approved|certified|"
        r"safe|suitable|usable)\b"
    ),
    re.compile(
        r"(?<!不)(?<!未)(?:达到|具备|符合|通过|适用于|可用于|保证|证明)"
        r"[^\n。！？;；,，、:：]{0,24}"
        r"(?:生产|医疗|临床|医学)(?:级|环境|可用|安全|部署|使用|正确性|能力|就绪|认证)?"
    ),
    re.compile(
        r"(?<!不)(?:生产级|医疗级|临床级)"
        r"(?:系统|平台|助手|服务)?(?:可用|就绪|安全|验证|认证)"
    ),
    re.compile(
        r"(?<!不)(?:这是(?:一个)?|(?:本|该)(?:项目|系统|助手)(?:是|为))"
        r"[^\n。！？;；,，、:：]{0,12}"
        r"(?:生产级|医疗级|临床级)"
        r"[^\n。！？;；,，、:：]{0,16}"
        r"(?:系统|平台|助手|问答)"
    ),
    re.compile(
        r"(?<!不)(?:生产级|医疗级|临床级)"
        r"[^\n。！？;；,，、:：]{0,20}"
        r"(?:系统|平台|助手|服务)"
    ),
)

_RUNTIME_MULTI_AGENT_PATTERNS = (
    re.compile(
        r"(?i)(?:产品|系统|应用|助手|服务|请求|运行时|"
        r"product(?:\s+runtime)?|runtime|application|service)"
        r"[^\n。！？.!?;；,，、:：]{0,36}"
        r"(?:采用|使用|调用|启用|部署|运行|包含|协调|编排|由|通过|"
        r"uses?|adopts?|invokes?|runs?|orchestrates?|includes?|employs?)"
        r"[^\n。！？.!?;；,，、:：]{0,24}(?:多\s*Agent|多个\s*Agent|"
        r"multi[- ]agent|multiple\s+agents)"
    ),
    re.compile(
        r"(?i)(?:多\s*Agent|multi[- ]agent|multiple\s+agents)"
        r"[^\n。！？.!?;；,，、:：]{0,24}(?:运行时|runtime)"
    ),
    re.compile(
        r"(?i)(?:多个\s*Agent|multiple\s+agents)[^\n。！？.!?;；,，、:：]{0,20}"
        r"(?:协同|协作|编排|完成|运行|collaborat\w*|orchestrat\w*|"
        r"work\s+together|run)"
    ),
)

_PROVIDER_COMPATIBILITY_PATTERNS = (
    re.compile(
        r"(?i)(?:支持|兼容|接入|集成|调用|使用|启用|采用|适配|配置|"
        r"fallback|adapter|integrat\w*|support\w*|compatib\w*|"
        r"use\w*|call\w*|adopt\w*)"
        r"[^\n。！？.!?,，、:：;；]{0,24}(?:MiniMax|TaoToken)"
    ),
    re.compile(
        r"(?i)(?:MiniMax|TaoToken)[^\n。！？.!?,，、:：;；]{0,24}"
        r"(?:支持|兼容|接入|集成|调用|使用|启用|采用|适配|配置|"
        r"fallback|adapter|integrat\w*|support\w*|compatib\w*)"
    ),
    re.compile(
        r"(?i)(?:MiniMax|TaoToken)[^\n。！？.!?,，、:：;；]{0,16}"
        r"(?:作为|as)\s*(?:备用|备选|fallback|backup)\s*(?:Provider|"
        r"提供商|供应商)"
    ),
)

_DEMO_PROVIDER_OVERCLAIM_PATTERNS = (
    re.compile(
        r"(?i)(?<![A-Za-z0-9])Demo(?![A-Za-z0-9])[^\n。！？.!?,，、:：;；]{0,72}"
        r"(?:读取|读|使用|调用|需要|访问|"
        r"(?<![A-Za-z0-9])reads?(?![A-Za-z0-9])|"
        r"(?<![A-Za-z0-9])uses?(?![A-Za-z0-9])|"
        r"(?<![A-Za-z0-9])calls?(?![A-Za-z0-9])|"
        r"(?<![A-Za-z0-9])requires?(?![A-Za-z0-9])|"
        r"(?<![A-Za-z0-9])accesses?(?![A-Za-z0-9]))"
        r"[^\n。！？.!?,，、:：;；]{0,24}"
        r"(?:Key|DeepSeek|SiliconFlow|Cloud|云端|网络|外网|network|internet)"
    ),
)
_CLOUD_WITHOUT_KEY_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])Cloud(?![A-Za-z0-9])[^\n。！？.!?,，、:：;；]{0,80}"
    r"(?:无需|不需要|不用|不读取|不使用|without|does not need|no)"
    r"[^\n。！？.!?,，、:：;；]{0,16}(?:Key|key|凭据|credential)"
)

_PRIVATE_ABSOLUTE_PATH_RE = re.compile(
    r"(?ix)(?<![A-Za-z0-9])(?:"
    r"[A-Z]:[\\/](?:[^\\/\r\n`*?]+[\\/])*[^\\/\r\n`*?]*"
    r"|\\\\[^\\/\r\n`*?]+[\\/][^\\/\r\n`*?]+"
    r"|/(?:users|home|root|private|workspace|var|tmp|mnt|opt|srv)"
    r"(?:/[^\s`<>()]+)*"
    r")"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?ix)\b(?:"
    r"[a-z0-9_. -]*(?:api[_ -]?key|access[_ -]?token)"
    r"|(?:deepseek|siliconflow|minimax|taotoken)[_ -]?(?:key|token)"
    r"|authorization|bearer|secret(?:[_ -]?key)?|password|token"
    r")\s*(?:=|:)\s*"
    r"[\"']?(?P<value>[A-Za-z0-9][A-Za-z0-9._~+/=-]{11,})[\"']?"
)
_BEARER_VALUE_RE = re.compile(
    r"(?i)\b(?:authorization\s*[:=]\s*)?bearer\s+"
    r"[A-Za-z0-9._~+/=-]{16,}\b"
)
_TOKEN_RE = re.compile(r"(?i)\b(?:sk|pk|sf|mm)-[A-Za-z0-9_-]{16,}\b")
_SAFE_SECRET_VALUE_RE = re.compile(
    r"(?i)^(?:placeholder|example|empty|none|null|missing|not[_ -]?set|"
    r"your[_ -]?key|own[_ -]?key|replace[_ -]?me|redacted|自备|本机|留空|未填写)"
)


def markdown_links(text: str) -> tuple[str, ...]:
    """Return relative Markdown links and image targets."""

    return tuple(
        link
        for link in _MARKDOWN_LINK_RE.findall(text)
        if not link.startswith(("http://", "https://", "mailto:"))
    )


def resolve_markdown_link(source: Path, link: str) -> Path:
    """Resolve a link target while ignoring an optional heading fragment."""

    target = link.split("#", 1)[0].strip()
    return source if not target else (source.parent / target).resolve()


def read_public_markdown() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)


def test_public_docs_describe_task16b_data_minimization():
    evaluation = (PROJECT_ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
    assert "切片后按需发送" in evaluation
    assert "不把整份 Word 或全部 1,114 个切片" in evaluation
    assert "Task 16B" in evaluation
    assert "真实题目" not in evaluation


def _readme_top_level_headings(text: str) -> tuple[str, ...]:
    headings: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and re.fullmatch(r"#{1,2}\s+.+", line.strip()):
            headings.append(line.strip())
    return tuple(headings)


def _fenced_bash_blocks(text: str) -> tuple[str, ...]:
    return tuple(
        re.findall(r"```(?:bash|sh)\s*\n(.*?)```", text, flags=re.DOTALL)
    )


def _section_between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index + len(start))
    return text[start_index:end_index]


def _has_nearby_marker(
    text: str,
    match: re.Match[str],
    marker: re.Pattern[str],
    *,
    radius: int = 16,
) -> bool:
    left = max(0, match.start() - radius)
    right = min(len(text), match.end() + radius)
    prefix = text[left:match.start()]
    previous_boundary = list(_CLAUSE_BOUNDARY_RE.finditer(prefix))
    if previous_boundary:
        prefix = prefix[previous_boundary[-1].end() :]
    suffix = text[match.end() : right]
    next_boundary = _CLAUSE_BOUNDARY_RE.search(suffix)
    if next_boundary:
        suffix = suffix[: next_boundary.start()]
    return bool(marker.search(match.group()) or marker.search(prefix) or marker.search(suffix))


def _has_affirmative_claim(
    text: str,
    patterns: tuple[re.Pattern[str], ...],
    *,
    skip_future: bool = False,
) -> bool:
    for pattern in patterns:
        for match in pattern.finditer(text):
            if _has_nearby_marker(text, match, _NEGATION_RE):
                continue
            if skip_future and _has_nearby_marker(text, match, _FUTURE_RE):
                continue
            return True
    return False


def _without_urls(text: str) -> str:
    return re.sub(r"(?i)\b(?:https?|ftp)://[^\s)]+", " ", text)


def _has_private_absolute_path(text: str) -> bool:
    return bool(_PRIVATE_ABSOLUTE_PATH_RE.search(_without_urls(text)))


def _has_credential_material(text: str) -> bool:
    for match in _SECRET_ASSIGNMENT_RE.finditer(text):
        value = match.group("value").strip("\"'")
        if not _SAFE_SECRET_VALUE_RE.match(value):
            return True
    return bool(_BEARER_VALUE_RE.search(text) or _TOKEN_RE.search(text))


def test_all_public_markdown_files_exist():
    assert all(path.is_file() for path in PUBLIC_DOCS)


def test_v1_1_governance_docs_distinguish_engineering_evidence_from_business_readiness():
    """Keep v1.1's honest claims and scenario contracts from silently regressing."""

    assert all(path.is_file() for path in PROJECT_DOCS)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PROJECT_DOCS)

    assert "可见引用不等于答案正确" in combined
    assert "资料有版本和来源定位" in combined
    assert "安全门先于回答率" in combined
    assert "不等于医学准确率认证" in combined
    assert "不把它包装成已经上线的临床系统" in combined


def test_readme_follows_the_approved_top_level_heading_sequence():
    text = README.read_text(encoding="utf-8")
    assert _readme_top_level_headings(text) == EXPECTED_README_HEADINGS


def test_public_readme_leads_with_reference_implementation_not_demo_only():
    readme = README.read_text(encoding="utf-8")

    assert "证据约束的医疗知识 RAG 参考实现" in readme
    assert "30 秒了解项目" in readme
    assert "工程亮点" in readme


def test_release_evidence_does_not_claim_an_unverified_tag_or_remote_release():
    evidence = (PROJECT_ROOT / "docs" / "release-evidence.md").read_text(
        encoding="utf-8"
    )

    assert "远程状态需在发布前核验" in evidence
    assert "未创建或推送 Git tag" in evidence


def test_release_evidence_names_the_public_validator_as_the_release_security_gate():
    evidence = (PROJECT_ROOT / "docs" / "release-evidence.md").read_text(
        encoding="utf-8"
    )

    assert "`validate_public_release.py`" in evidence
    assert "通用 `security_scan.py`" in evidence
    assert "不作为公开副本的通过结论" in evidence


def test_all_public_markdown_relative_links_resolve():
    broken: list[str] = []
    for source in PUBLIC_DOCS:
        for link in markdown_links(source.read_text(encoding="utf-8")):
            if not resolve_markdown_link(source, link).exists():
                broken.append(f"{source.name}: {link}")
    assert not broken, "broken public-doc links: " + ", ".join(broken)
    evaluation = (PROJECT_ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
    assert "文档契约" in evaluation
    assert "`12 passed`" not in evaluation
    assert "`13 passed`" not in evaluation


def test_readme_has_separate_local_demo_and_cloud_terminal_commands():
    text = README.read_text(encoding="utf-8")
    demo = _section_between(text, "### macOS/Linux 手动启动", "### 自备 Key 的 Cloud 模式")
    cloud = _section_between(text, "### 自备 Key 的 Cloud 模式", "## 已实现功能")
    demo_blocks = _fenced_bash_blocks(demo)
    cloud_blocks = _fenced_bash_blocks(cloud)

    assert len(demo_blocks) >= 2
    assert any("cd backend" in block and "uvicorn" in block for block in demo_blocks)
    assert any("cd frontend" in block and "npm run dev" in block for block in demo_blocks)
    assert all(not ("uvicorn" in block and "npm run dev" in block) for block in demo_blocks)
    assert len(cloud_blocks) >= 2
    assert any("cd backend" in block and "APP_RUNTIME_MODE=cloud" in block for block in cloud_blocks)
    assert any("cd frontend" in block and "npm run dev" in block for block in cloud_blocks)
    server_blocks = tuple(
        block
        for block in demo_blocks + cloud_blocks
        if "uvicorn" in block or "npm run dev" in block
    )
    assert all(
        "nohup" not in block and not re.search(r"(?m)&\s*$", block)
        for block in server_blocks
    )


def test_public_docs_state_the_synthetic_demo_and_explicit_cloud_boundary():
    combined = read_public_markdown()
    assert re.search(r"(?:完全|原创)虚构", combined)
    assert re.search(r"Demo[^\n。！？]{0,90}(?:无需|不读取|不读)[^\n。！？]{0,24}Key", combined)
    assert re.search(r"Demo[^\n。！？]{0,90}(?:无需网络|不访问外网|不访问网络)", combined)
    assert re.search(r"Cloud[^\n。！？]{0,110}(?:显式|自备)[^\n。！？]{0,24}Key", combined)
    assert not _has_affirmative_claim(combined, _DEMO_PROVIDER_OVERCLAIM_PATTERNS)
    assert not _CLOUD_WITHOUT_KEY_PATTERN.search(combined)


def test_public_docs_explain_literal_vs_semantic_metrics_without_claiming_results():
    evaluation = (PROJECT_ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")

    assert "逐字覆盖" in evaluation
    assert "语义评测基础设施" in evaluation
    assert "语义基线已受控执行一次，但尚不稳定" in evaluation
    assert "不能作为稳定语义分数" in evaluation
    assert "真实题目" not in evaluation


def test_public_metrics_require_denominator_and_limitations():
    """Keep aggregate observations auditable and stop business overclaiming."""

    evaluation = (PROJECT_ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
    assert "20/30" in evaluation
    assert "不是医学准确率" in evaluation
    assert "真实采纳率" not in evaluation
    assert "临床改善已验证" not in evaluation
    assert "ROI 已验证" not in evaluation
    assert "## 指标字典（分子、分母、置信区间与限制）" in evaluation
    assert "C1/C2 与 9F-B 使用不同评测协议，不能横向比较" in evaluation


def test_public_manifest_allows_governance_files_but_not_private_semantic_runner():
    manifest = PUBLIC_RELEASE_MANIFEST

    assert "CHANGELOG.md" in manifest.files
    assert ".github/PULL_REQUEST_TEMPLATE.md" in manifest.files
    assert "docs/release-evidence.md" in manifest.files
    assert "scripts/task9f_semantic_evaluation.py" not in manifest.files
    assert "backend/tests/evaluation/test_task9f_semantic_script.py" not in manifest.files


def test_public_manifest_excludes_private_phase2a_runner_tests():
    manifest = PUBLIC_RELEASE_MANIFEST

    assert "scripts/v2a_evaluate.py" not in manifest.files
    assert "backend/tests/evaluation/test_v2a_preflight.py" not in manifest.files
    assert "backend/tests/evaluation/test_v2a_evaluate.py" not in manifest.files


def test_public_manifest_excludes_real_data_heading_audit():
    manifest = PUBLIC_RELEASE_MANIFEST

    assert "scripts/heading_extraction_audit.py" not in manifest.files
    assert (
        "backend/tests/ingestion/test_heading_extraction_audit.py"
        in manifest.excluded_files
    )


def test_public_docs_do_not_make_positive_production_or_runtime_agent_claims():
    combined = read_public_markdown()
    assert not _has_affirmative_claim(combined, _PRODUCTION_CLAIM_PATTERNS)
    assert not _has_affirmative_claim(combined, _RUNTIME_MULTI_AGENT_PATTERNS, skip_future=True)


def test_public_docs_do_not_claim_unapproved_provider_compatibility():
    combined = read_public_markdown()
    assert not _has_affirmative_claim(
        combined, _PROVIDER_COMPATIBILITY_PATTERNS, skip_future=True
    )


def test_public_docs_do_not_contain_private_paths_or_credential_material():
    combined = read_public_markdown()
    assert not _has_private_absolute_path(combined)
    assert not _has_credential_material(combined)


def test_claim_guards_catch_equivalent_positive_overclaim_canaries():
    production = "系统已达到" + "生产级医疗可用性。"
    production_variant = "这是一个" + "生产级医疗问答系统。"
    production_english = "production-grade medical system"
    runtime_agents = "产品运行时采用" + "多 Agent 协作。"
    runtime_variant = "系统由多个 Agent 协同完成。"
    compatibility = "Cloud 兼容 " + "MiniMax/TaoToken。"
    compatibility_variant = "MiniMax/TaoToken 作为备用 Provider。"
    assert _has_affirmative_claim(production, _PRODUCTION_CLAIM_PATTERNS)
    assert _has_affirmative_claim(production_variant, _PRODUCTION_CLAIM_PATTERNS)
    assert _has_affirmative_claim(production_english, _PRODUCTION_CLAIM_PATTERNS)
    assert _has_affirmative_claim(runtime_agents, _RUNTIME_MULTI_AGENT_PATTERNS)
    assert _has_affirmative_claim(runtime_variant, _RUNTIME_MULTI_AGENT_PATTERNS)
    assert _has_affirmative_claim(compatibility, _PROVIDER_COMPATIBILITY_PATTERNS)
    assert _has_affirmative_claim(compatibility_variant, _PROVIDER_COMPATIBILITY_PATTERNS)
    mixed_limit_and_claim = "当前不是生产系统，但" + runtime_agents
    assert _has_affirmative_claim(mixed_limit_and_claim, _RUNTIME_MULTI_AGENT_PATTERNS)


def test_claim_guards_allow_negated_limits_and_future_candidates():
    safe_limits = (
        "这不是生产级医疗系统，也不能用于临床。"
        "产品运行时不采用多 Agent。"
        "MiniMax/TaoToken 暂不接入，属于 v2.0 候选。"
    )
    assert not _has_affirmative_claim(safe_limits, _PRODUCTION_CLAIM_PATTERNS)
    assert not _has_affirmative_claim(
        safe_limits, _RUNTIME_MULTI_AGENT_PATTERNS, skip_future=True
    )
    assert not _has_affirmative_claim(
        safe_limits, _PROVIDER_COMPATIBILITY_PATTERNS, skip_future=True
    )


def test_boundary_guards_catch_cross_mode_overclaim_canaries():
    demo_uses_cloud = "Demo 模式使用 DeepSeek Key 并访问外网。"
    cloud_without_key = "Cloud 模式无需 Key 即可调用 Provider。"
    assert _has_affirmative_claim(demo_uses_cloud, _DEMO_PROVIDER_OVERCLAIM_PATTERNS)
    assert _CLOUD_WITHOUT_KEY_PATTERN.search(cloud_without_key)


def test_private_text_guards_catch_path_and_credential_canaries():
    windows_separator = chr(92)
    windows_path = "C:" + windows_separator + "Users" + windows_separator + "private" + windows_separator + "answer.sqlite3"
    posix_path = "/" + "home/private/answer.sqlite3"
    credential = "DEEPSEEK_API_KEY=" + "sk-" + ("x" * 24)
    bearer = "Authorization: " + "Bear" + "er " + ("x" * 24)
    assert _has_private_absolute_path(windows_path)
    assert _has_private_absolute_path(posix_path)
    assert _has_credential_material(credential)
    assert _has_credential_material(bearer)
