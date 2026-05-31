"""Transformer（转换器/分片）：将已解析文档内容按策略切分为 Chunk (Req 6)。

本模块是知识库入库管线的第二环：接收 :class:`~ghost_agent.core.loader.ParseResult`
（Loader 的输出），依据生效的分片策略将统一纯文本切分为一个或多个
:class:`~ghost_agent.models.chunk.Chunk`，并在内容为空、策略不可应用等情形下
按需终止处理。

设计核心 —— **基于切片的装箱算法（slice-based packer）**：

设计的关键在于"让三条正确性属性按构造（by construction）成立"，而非靠事后校验：

1. **规范字符流 + 软边界提示**：先依据策略把内容重建为单一字符流
   ``stream``，并给出一组"宜切分"的整型边界位置 ``boundaries``（段落/标题起始、
   句末标点、换行等）。边界永远包含 0 与 ``len(stream)``。
2. **装箱**：在 ``stream`` 上从左到右切出半开区间 ``[start, end)``：
   * 当剩余长度仍 > ``max_length`` 时，必须切出一个"非最后" Chunk —— 在窗口
     ``[cursor+min_length, cursor+max_length]`` 内取**最大**的边界；若窗口内无
     边界，则在 ``cursor+max_length`` 处硬切。两种情形得到的长度都落在
     ``[min_length, max_length]``（含端点），从而 **Property 1** 成立。
   * 循环结束后剩余 ``[cursor, len(stream))`` 即"最后一个" Chunk（豁免下界）。
3. **元数据**：切片按顺序赋予连续序号 ``seq=0..n-1``、来源文件标识与起止位置；
   切片左闭右开且首尾相接，故 ``start_offset <= end_offset`` 且起止位置随序号
   单调不减 —— **Property 3** 成立。
4. **二次切分**：装箱完成后，对任一长度超过 ``max_input_length`` 的切片做硬字符
   切分为若干 ``<= max_input_length`` 的子切片（``parent_chunk_id`` 指向同一父
   标识），并在最终列表上重排 ``seq`` 保持连续 —— **Property 2** 成立，且与装箱
   参数无关（即便 ``max_length > max_input_length`` 也成立）。

注意：``start_offset``/``end_offset`` 是相对于重建后的规范流 ``stream`` 的位置
（而非原始 ``ParseResult.text`` 的位置）。这样既保证位置精确（切片即文本），又
天然满足序号-位置单调性。

对应需求：6.1（长度边界）、6.2（三策略 + 默认策略）、6.3（元数据与序号位置）、
6.4（超长二次切分）、6.5（移交 Indexer，由上层编排）、6.6（内容为空不生成）、
6.7（策略不可应用/配置非法则终止并返回错误）。
"""

from __future__ import annotations

import bisect
import uuid
from enum import Enum

from ghost_agent.config import get_settings
from ghost_agent.core.loader import ParseResult
from ghost_agent.models.chunk import Chunk
from ghost_agent.models.errors import SplitFailedError

__all__ = ["ChunkStrategy", "Transformer"]


class ChunkStrategy(str, Enum):
    """受支持的分片策略 (Req 6.2)。"""

    BY_HEADING = "BY_HEADING"
    BY_PARAGRAPH = "BY_PARAGRAPH"
    BY_SEMANTIC = "BY_SEMANTIC"


#: 段落/标题单元之间使用的分隔符。
_UNIT_SEPARATOR = "\n\n"
#: BY_HEADING 策略下，单个区块内标题与段落之间的分隔符。
_INTRA_SECTION_SEPARATOR = "\n"
#: BY_SEMANTIC 策略下视为句末的标点集合（中英文混合）。
_SENTENCE_END_CHARS = frozenset("。！？!?.;；")
#: 句末标点后可并入边界的"行内空白"。
_INLINE_WHITESPACE = frozenset(" \t")


class Transformer:
    """文档分片器：将 :class:`ParseResult` 切分为 :class:`Chunk` 列表 (Req 6)。

    构造参数缺省时取自全局 :class:`~ghost_agent.config.Settings`
    （``chunk_min_length``、``chunk_max_length``、``embedding_max_input_length``、
    ``default_chunk_strategy``）。

    长度参数的合法性（``0 < min_length <= max_length`` 且 ``max_input_length > 0``）
    在 :meth:`split` 执行期校验，非法时抛出 :class:`SplitFailedError` (Req 6.7)。
    刻意 **不** 要求 ``max_length <= max_input_length``，以便超长内容触发二次切分
    （Req 6.4 / Property 2）。
    """

    def __init__(
        self,
        *,
        min_length: int | None = None,
        max_length: int | None = None,
        max_input_length: int | None = None,
        default_strategy: "ChunkStrategy | str | None" = None,
    ) -> None:
        need_settings = (
            min_length is None
            or max_length is None
            or max_input_length is None
            or default_strategy is None
        )
        settings = get_settings() if need_settings else None
        self._min_length: int = (
            min_length if min_length is not None else settings.chunk_min_length
        )
        self._max_length: int = (
            max_length if max_length is not None else settings.chunk_max_length
        )
        self._max_input_length: int = (
            max_input_length
            if max_input_length is not None
            else settings.embedding_max_input_length
        )
        self._default_strategy: "ChunkStrategy | str" = (
            default_strategy
            if default_strategy is not None
            else settings.default_chunk_strategy
        )

    # ------------------------------- 公共 API -------------------------------- #
    def split(
        self,
        parse_result: ParseResult,
        strategy: "ChunkStrategy | str | None" = None,
    ) -> list[Chunk]:
        """将解析结果切分为 Chunk 列表。

        Args:
            parse_result: Loader 输出的统一纯文本 + 结构化区块 + 元数据。
            strategy: 显式分片策略；为 ``None`` 时采用配置默认策略 (Req 6.2)。

        Returns:
            按序号 0..n-1 排列的 :class:`Chunk` 列表；内容为空时返回 ``[]``
            （不生成 Chunk、不移交 Indexer，Req 6.6）。

        Raises:
            SplitFailedError: 长度配置非法，或策略不可应用（含未知策略），Req 6.7。
        """
        self._validate_config()
        resolved = self._resolve_strategy(strategy)

        stream, boundaries = self._build_stream(parse_result, resolved)
        # Req 6.6：内容为空 → 不生成 Chunk、不移交 Indexer。
        if not stream.strip():
            return []

        slices = self._pack(stream, boundaries)
        # 防御性：若最后一个切片为纯空白且存在前序切片，并入前序，避免产出
        # 仅含分隔符的尾部 Chunk（不影响属性，仅为整洁）。
        slices = self._merge_trailing_whitespace(stream, slices)

        source_file_id = parse_result.meta.source_file_id
        return self._materialize(stream, slices, source_file_id)

    # ------------------------------- 配置/策略 ------------------------------- #
    def _validate_config(self) -> None:
        """执行期校验长度配置 (Req 6.7)。"""
        if self._min_length <= 0:
            raise SplitFailedError(
                f"分片配置非法：min_length 必须为正整数，实际 {self._min_length}"
            )
        if self._min_length > self._max_length:
            raise SplitFailedError(
                "分片配置非法：min_length 不得大于 max_length："
                f"min={self._min_length}, max={self._max_length}"
            )
        if self._max_input_length <= 0:
            raise SplitFailedError(
                "分片配置非法：max_input_length 必须为正整数，"
                f"实际 {self._max_input_length}"
            )

    def _resolve_strategy(
        self, strategy: "ChunkStrategy | str | None"
    ) -> ChunkStrategy:
        """解析生效策略；未指定取默认，未知策略字符串视为不可应用 (Req 6.2/6.7)。"""
        candidate = strategy if strategy is not None else self._default_strategy
        if isinstance(candidate, ChunkStrategy):
            return candidate
        try:
            return ChunkStrategy(str(candidate))
        except ValueError as exc:
            supported = ", ".join(s.value for s in ChunkStrategy)
            raise SplitFailedError(
                f"不支持的分片策略: {candidate!r}；受支持策略: [{supported}]"
            ) from exc

    # ----------------------------- 规范流构建 -------------------------------- #
    def _build_stream(
        self, parse_result: ParseResult, strategy: ChunkStrategy
    ) -> tuple[str, list[int]]:
        """按策略重建规范字符流并给出软边界位置集合。"""
        if strategy is ChunkStrategy.BY_SEMANTIC:
            text = self._raw_text(parse_result)
            return text, self._semantic_boundaries(text)

        units = self._build_units(parse_result, strategy)
        if not units:
            # 回退：无可用单元时退化为整段原始文本（仍可能为空，由上层判空）。
            text = self._raw_text(parse_result)
            units = [text] if text.strip() else []
        stream = _UNIT_SEPARATOR.join(units)
        return stream, self._unit_boundaries(units, _UNIT_SEPARATOR)

    @staticmethod
    def _raw_text(parse_result: ParseResult) -> str:
        """取统一纯文本；为空时由结构化区块重建。"""
        if parse_result.text and parse_result.text.strip():
            return parse_result.text
        parts: list[str] = []
        for section in parse_result.sections:
            if section.title:
                parts.append(section.title)
            parts.extend(p for p in section.paragraphs if p.strip())
        return _UNIT_SEPARATOR.join(parts)

    @staticmethod
    def _build_units(parse_result: ParseResult, strategy: ChunkStrategy) -> list[str]:
        """按策略将结构化区块转换为"切分单元"列表（每个单元已去除首尾空白）。"""
        units: list[str] = []
        if strategy is ChunkStrategy.BY_PARAGRAPH:
            for section in parse_result.sections:
                for paragraph in section.paragraphs:
                    stripped = paragraph.strip()
                    if stripped:
                        units.append(stripped)
        elif strategy is ChunkStrategy.BY_HEADING:
            for section in parse_result.sections:
                parts: list[str] = []
                if section.title and section.title.strip():
                    parts.append(section.title.strip())
                parts.extend(p.strip() for p in section.paragraphs if p.strip())
                unit = _INTRA_SECTION_SEPARATOR.join(parts).strip()
                if unit:
                    units.append(unit)
        return units

    @staticmethod
    def _unit_boundaries(units: list[str], separator: str) -> list[int]:
        """单元起始处（及流首尾）作为软边界，保证切片以真实内容开头。"""
        bounds = {0}
        pos = 0
        for idx, unit in enumerate(units):
            if idx > 0:
                pos += len(separator)
            bounds.add(pos)  # 当前单元起始位置
            pos += len(unit)
        bounds.add(pos)  # == len(stream)
        return sorted(bounds)

    @staticmethod
    def _semantic_boundaries(text: str) -> list[int]:
        """句末标点（含其后行内空白）与换行处作为软边界，确定性、无 LLM/随机。"""
        n = len(text)
        bounds = {0, n}
        for i, ch in enumerate(text):
            if ch == "\n":
                bounds.add(i + 1)
            elif ch in _SENTENCE_END_CHARS:
                j = i + 1
                while j < n and text[j] in _INLINE_WHITESPACE:
                    j += 1
                bounds.add(j)
        return sorted(bounds)

    # ------------------------------- 装箱算法 -------------------------------- #
    def _pack(self, stream: str, boundaries: list[int]) -> list[tuple[int, int]]:
        """在 ``stream`` 上切出半开区间切片列表，保证 Property 1。

        不变式：
        * 每个"非最后"切片长度 ∈ [min_length, max_length]（含端点）。
        * 切片首尾相接、左闭右开，覆盖整个 [0, len(stream))。
        * cursor 每轮严格前进 >= min_length，循环必然终止。
        """
        n = len(stream)
        min_len = self._min_length
        max_len = self._max_length
        slices: list[tuple[int, int]] = []
        cursor = 0
        while n - cursor > max_len:
            low = cursor + min_len
            high = cursor + max_len
            cut = _largest_boundary(boundaries, low, high)
            if cut is None:
                cut = high  # 窗口内无软边界 → 硬切，长度恰为 max_len ∈ [min,max]
            slices.append((cursor, cut))
            cursor = cut
        # 剩余部分为"最后一个"切片（豁免下界）；非空才产出。
        if cursor < n:
            slices.append((cursor, n))
        return slices

    @staticmethod
    def _merge_trailing_whitespace(
        stream: str, slices: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """若末切片为纯空白且存在前序切片，则并入前序切片（整洁性保障）。"""
        if len(slices) >= 2:
            last_start, last_end = slices[-1]
            if not stream[last_start:last_end].strip():
                prev_start, _ = slices[-2]
                slices = slices[:-2] + [(prev_start, last_end)]
        return slices

    # ------------------------------- 物化 Chunk ------------------------------ #
    def _materialize(
        self, stream: str, slices: list[tuple[int, int]], source_file_id: str
    ) -> list[Chunk]:
        """将切片物化为 Chunk，并对超长切片执行二次切分（Req 6.3/6.4）。

        不变式：
        * 最终序号 seq 连续递增 0..n-1（Property 3）。
        * 每个最终 Chunk 文本长度 <= max_input_length（Property 2）。
        * 起止位置随序号单调不减且 start_offset <= end_offset（Property 3）。
        """
        max_input = self._max_input_length
        chunks: list[Chunk] = []
        seq = 0
        for start, end in slices:
            if end - start <= max_input:
                chunks.append(
                    Chunk(
                        source_file_id=source_file_id,
                        seq=seq,
                        start_offset=start,
                        end_offset=end,
                        text=stream[start:end],
                    )
                )
                seq += 1
                continue

            # Req 6.4：超长切片二次切分为不超过 max_input 的子 Chunk。
            parent_id = str(uuid.uuid4())
            sub_cursor = start
            while sub_cursor < end:
                sub_end = min(sub_cursor + max_input, end)
                chunks.append(
                    Chunk(
                        source_file_id=source_file_id,
                        seq=seq,
                        start_offset=sub_cursor,
                        end_offset=sub_end,
                        text=stream[sub_cursor:sub_end],
                        parent_chunk_id=parent_id,
                    )
                )
                seq += 1
                sub_cursor = sub_end
        return chunks


def _largest_boundary(boundaries: list[int], low: int, high: int) -> int | None:
    """返回排序边界列表中满足 ``low <= b <= high`` 的最大边界；无则 ``None``。"""
    idx = bisect.bisect_right(boundaries, high) - 1
    if idx >= 0 and boundaries[idx] >= low:
        return boundaries[idx]
    return None
