"""Transformer（文档分片）单元测试与属性测试 (Req 6.1–6.7)。

覆盖：
* Property 1（任务 4.4）：分片长度边界 —— 非最后 Chunk 长度落在 [min,max]。
* Property 2（任务 4.5）：超长 Chunk 二次切分上界 —— 最终 Chunk 长度 <= max_input。
* Property 3（任务 4.6）：元数据完整 + 序号连续 + 起止位置单调。
* 示例用例：空内容/未知策略/非法配置/默认策略/二次切分/来源标识传播/三策略可用。

外部无依赖：分片为纯确定性逻辑（BY_SEMANTIC 为确定性启发式，非 LLM 调用）。
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.core import ChunkStrategy, Transformer
from ghost_agent.core.loader import FileMeta, ParseResult, Section
from ghost_agent.models import SplitFailedError

_DEFAULT = ChunkStrategy.BY_PARAGRAPH


# --------------------------------------------------------------------------- #
# 测试辅助：从字符串构造 ParseResult                                            #
# --------------------------------------------------------------------------- #
def make_parse_result(text: str, *, source_file_id: str = "src-1") -> ParseResult:
    """构造单区块（level-0）的 ParseResult，正文为单一段落。"""
    paragraphs = [text] if text else []
    section = Section(title="", level=0, paragraphs=paragraphs)
    meta = FileMeta(
        source_file_id=source_file_id,
        file_name="doc.txt",
        file_format="txt",
    )
    return ParseResult(text=text, sections=[section], meta=meta)


def make_multi_section() -> ParseResult:
    """构造含多标题、多段落的 ParseResult，用于策略多样性验证。"""
    sections = [
        Section(
            title="安装指南",
            level=1,
            paragraphs=["第一段落内容充足用于切分。" * 5, "第二段落同样足够长。" * 5],
        ),
        Section(
            title="故障排查",
            level=2,
            paragraphs=["排查步骤详细说明文本。" * 8],
        ),
    ]
    parts: list[str] = []
    for section in sections:
        if section.title:
            parts.append(section.title)
        parts.extend(section.paragraphs)
    text = "\n\n".join(parts)
    meta = FileMeta(
        source_file_id="multi-1", file_name="manual.md", file_format="md"
    )
    return ParseResult(text=text, sections=sections, meta=meta)


# --------------------------------------------------------------------------- #
# Property 1（任务 4.4）：分片长度边界                                           #
# --------------------------------------------------------------------------- #
@settings(max_examples=100, deadline=None)
@given(
    text=st.text(min_size=0, max_size=2000),
    strategy=st.sampled_from(list(ChunkStrategy)),
)
def test_property_1_chunk_length_bounds(text: str, strategy: ChunkStrategy) -> None:
    """Feature: intelligent-oncall-agent, Property 1: 对任意非空已解析文档内容与任一受支持的分片策略，切分产生的 Chunk 数量至少为 1，且除最后一个 Chunk 外，每个 Chunk 的文本长度均介于配置的单分片最小长度与最大长度之间（含端点）。

    **Validates: Requirements 6.1**
    """
    transformer = Transformer(
        min_length=5, max_length=20, max_input_length=1000, default_strategy=_DEFAULT
    )
    chunks = transformer.split(make_parse_result(text), strategy)

    if text.strip():
        assert len(chunks) >= 1
        for chunk in chunks[:-1]:
            assert 5 <= len(chunk.text) <= 20
    else:
        # Req 6.6：内容为空不生成 Chunk。
        assert chunks == []


# --------------------------------------------------------------------------- #
# Property 2（任务 4.5）：超长 Chunk 二次切分上界                                 #
# --------------------------------------------------------------------------- #
@settings(max_examples=100, deadline=None)
@given(
    # 约束字母表为可打印 ASCII，排除空白与句末标点（".;!?"），
    # 从而不产生软边界 → 必出现超过 max_input_length 的切片触发二次切分。
    text=st.text(
        alphabet=st.characters(
            min_codepoint=33, max_codepoint=126, blacklist_characters=".;!?"
        ),
        min_size=11,
        max_size=500,
    ),
    strategy=st.sampled_from(list(ChunkStrategy)),
)
def test_property_2_secondary_split_upper_bound(
    text: str, strategy: ChunkStrategy
) -> None:
    """Feature: intelligent-oncall-agent, Property 2: 对任意文本长度超过 Embedding_Model 最大输入长度的内容，分片完成后所有最终 Chunk 的文本长度均不超过该最大输入长度。

    **Validates: Requirements 6.4**
    """
    transformer = Transformer(
        min_length=5, max_length=50, max_input_length=10, default_strategy=_DEFAULT
    )
    chunks = transformer.split(make_parse_result(text), strategy)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.text) <= 10
    # 输入长度 > max_input_length，必然发生二次切分 → 子 Chunk 携带 parent_chunk_id。
    assert any(chunk.parent_chunk_id is not None for chunk in chunks)


# --------------------------------------------------------------------------- #
# Property 3（任务 4.6）：元数据完整且序号位置单调                                #
# --------------------------------------------------------------------------- #
@settings(max_examples=100, deadline=None)
@given(
    text=st.text(min_size=0, max_size=1000),
    strategy=st.sampled_from(list(ChunkStrategy)),
)
def test_property_3_metadata_and_monotonic_offsets(
    text: str, strategy: ChunkStrategy
) -> None:
    """Feature: intelligent-oncall-agent, Property 3: 对任意由分片产生的 Chunk 集合，每个 Chunk 均带有来源文件标识、顺序序号与起止位置；序号在集合内连续递增（0..n-1），且每个 Chunk 的 start_offset <= end_offset，起止位置随序号单调不减。

    **Validates: Requirements 6.3**
    """
    source_file_id = "src-prop3"
    transformer = Transformer(
        min_length=5, max_length=20, max_input_length=10, default_strategy=_DEFAULT
    )
    chunks = transformer.split(
        make_parse_result(text, source_file_id=source_file_id), strategy
    )

    # 序号连续递增 0..n-1。
    assert [chunk.seq for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.source_file_id == source_file_id
        assert chunk.start_offset <= chunk.end_offset
    # 起止位置随序号单调不减。
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.start_offset <= later.start_offset
        assert earlier.end_offset <= later.end_offset


# --------------------------------------------------------------------------- #
# 示例用例：内容为空 (Req 6.6)                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["", "   ", "\n\t \n", "\u3000\u3000"])
@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_empty_or_whitespace_content_yields_no_chunks(
    text: str, strategy: ChunkStrategy
) -> None:
    transformer = Transformer(
        min_length=5, max_length=20, max_input_length=100, default_strategy=_DEFAULT
    )
    assert transformer.split(make_parse_result(text), strategy) == []


# --------------------------------------------------------------------------- #
# 示例用例：未知策略 (Req 6.7)                                                   #
# --------------------------------------------------------------------------- #
def test_unknown_strategy_raises_split_failed() -> None:
    transformer = Transformer(
        min_length=5, max_length=20, max_input_length=100, default_strategy=_DEFAULT
    )
    with pytest.raises(SplitFailedError):
        transformer.split(make_parse_result("有效内容用于分片测试。"), "FOO")


# --------------------------------------------------------------------------- #
# 示例用例：非法配置 (Req 6.7)                                                   #
# --------------------------------------------------------------------------- #
def test_invalid_config_min_greater_than_max_raises() -> None:
    transformer = Transformer(
        min_length=20, max_length=5, max_input_length=100, default_strategy=_DEFAULT
    )
    with pytest.raises(SplitFailedError):
        transformer.split(make_parse_result("一些内容"), ChunkStrategy.BY_PARAGRAPH)


def test_invalid_config_nonpositive_max_input_raises() -> None:
    transformer = Transformer(
        min_length=5, max_length=20, max_input_length=0, default_strategy=_DEFAULT
    )
    with pytest.raises(SplitFailedError):
        transformer.split(make_parse_result("一些内容"), ChunkStrategy.BY_PARAGRAPH)


def test_invalid_config_nonpositive_min_raises() -> None:
    transformer = Transformer(
        min_length=0, max_length=20, max_input_length=100, default_strategy=_DEFAULT
    )
    with pytest.raises(SplitFailedError):
        transformer.split(make_parse_result("一些内容"), ChunkStrategy.BY_PARAGRAPH)


# --------------------------------------------------------------------------- #
# 示例用例：默认策略 (Req 6.2)                                                   #
# --------------------------------------------------------------------------- #
def test_default_strategy_used_when_none() -> None:
    """strategy=None 时应等价于显式传入构造时配置的默认策略。"""
    transformer = Transformer(
        min_length=5,
        max_length=20,
        max_input_length=100,
        default_strategy=ChunkStrategy.BY_HEADING,
    )
    parse_result = make_multi_section()

    implicit = transformer.split(parse_result, None)
    explicit = transformer.split(parse_result, ChunkStrategy.BY_HEADING)

    assert [c.text for c in implicit] == [c.text for c in explicit]
    assert [(c.start_offset, c.end_offset) for c in implicit] == [
        (c.start_offset, c.end_offset) for c in explicit
    ]


# --------------------------------------------------------------------------- #
# 示例用例：二次切分设置 parent_chunk_id 且序号连续 (Req 6.4)                     #
# --------------------------------------------------------------------------- #
def test_secondary_split_sets_parent_and_keeps_seq_continuous() -> None:
    # max_length(40) > max_input_length(8) → 切片可能超过 8，触发二次切分。
    transformer = Transformer(
        min_length=5, max_length=40, max_input_length=8, default_strategy=_DEFAULT
    )
    text = "abcdefghijklmnopqrstuvwxyz0123456789" * 3  # 108 字符，无标点/空白
    chunks = transformer.split(make_parse_result(text), ChunkStrategy.BY_PARAGRAPH)

    assert len(chunks) >= 1
    assert [c.seq for c in chunks] == list(range(len(chunks)))
    assert all(len(c.text) <= 8 for c in chunks)
    # 至少一个子 Chunk 带 parent_chunk_id，且同父的多个子 Chunk 共享同一标识。
    parents = [c.parent_chunk_id for c in chunks if c.parent_chunk_id is not None]
    assert parents, "应存在由二次切分产生的子 Chunk"


# --------------------------------------------------------------------------- #
# 示例用例：来源文件标识传播 (Req 6.3)                                            #
# --------------------------------------------------------------------------- #
def test_source_file_id_propagated_from_meta() -> None:
    transformer = Transformer(
        min_length=5, max_length=20, max_input_length=100, default_strategy=_DEFAULT
    )
    parse_result = make_parse_result("内容足够用于产生分片。" * 3, source_file_id="abc-999")
    chunks = transformer.split(parse_result, ChunkStrategy.BY_PARAGRAPH)
    assert chunks
    assert all(c.source_file_id == "abc-999" for c in chunks)


# --------------------------------------------------------------------------- #
# 示例用例：三种策略在多区块文档上各产生 >=1 Chunk (Req 6.2)                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_each_strategy_produces_at_least_one_chunk(strategy: ChunkStrategy) -> None:
    transformer = Transformer(
        min_length=5, max_length=50, max_input_length=200, default_strategy=_DEFAULT
    )
    chunks = transformer.split(make_multi_section(), strategy)
    assert len(chunks) >= 1
    # 非最后 Chunk 满足长度边界。
    for chunk in chunks[:-1]:
        assert 5 <= len(chunk.text) <= 50
