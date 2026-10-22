"""Allow-listed knob catalog for framework/service-level tuning.

This is the safety boundary for agent-proposed changes: the planner can only
mutate knobs declared here, and each knob carries its applicability, cost and
risk.  Adding a knob is a reviewed code change, not an LLM output.

The catalog is seeded from public serving guides (launch flags, DP/TP, graph
mode, MTP), connector design docs (layerwise, KV pool, prefix affinity),
offload field notes (tier, threads, blocks_per_chunk) and quantization
trade-off notes. Each entry records where the prior came from.
"""
from __future__ import annotations

KNOBS = [
    {
        "key": "max_num_seqs",
        "label": "Decode 最大并发序列数",
        "type": "int",
        "unit": "seqs",
        "mutations": [{"kind": "scale", "factor": 1.25}, {"kind": "scale", "factor": 1.5}],
        "applies": {"hardware": ["A3", "A2", "A5", "910B", "950", "demo"],
                    "topology": ["1P1D", "PD", "colocated", "single"]},
        "cost": "restart",
        "risk": "low",
        "expected_pct": 4.0,
        "rationale": "提高 decode 批大小通常提升吞吐，但 TPOT 与 KV 显存上升；需同时看 tpot_p95 与 memory。",
        "guards": ["tpot_p95_ms", "memory_gb"],
        "source": "upstream serving guide",
    },
    {
        "key": "max_num_batched_tokens",
        "label": "单批 token 上限",
        "type": "int",
        "unit": "tokens",
        "mutations": [{"kind": "scale", "factor": 1.25}, {"kind": "scale", "factor": 1.5}],
        "applies": {"hardware": ["A3", "A2", "A5", "910B", "950", "demo"],
                    "topology": ["1P1D", "PD", "colocated", "single"]},
        "cost": "restart",
        "risk": "low",
        "expected_pct": 3.0,
        "rationale": "影响 prefill 批与 decode 调度粒度；过大易抬高 TTFT 与显存峰值。",
        "guards": ["ttft_p95_ms", "memory_gb"],
        "source": "upstream serving guide",
    },
    {
        "key": "async_scheduling",
        "label": "异步调度",
        "type": "bool",
        "mutations": [{"kind": "toggle"}],
        "applies": {"hardware": ["A3", "A2", "A5", "950"], "topology": ["1P1D", "PD", "colocated"]},
        "cost": "restart",
        "risk": "medium",
        "expected_pct": 2.0,
        "rationale": "官方 decode 脚本常用；对短请求/高并发可能改善气泡，但需验证输出一致性。",
        "guards": ["quality"],
        "source": "upstream serving guide",
    },
    {
        "key": "graph_mode",
        "label": "NPU 图捕获模式",
        "type": "enum",
        "values": ["FULL_DECODE_ONLY", "FULL", "PIECEWISE"],
        "mutations": [{"kind": "enum"}],
        "applies": {"hardware": ["A3", "A2", "A5", "950"], "topology": ["1P1D", "PD", "colocated"]},
        "cost": "restart",
        "risk": "medium",
        "expected_pct": 3.0,
        "rationale": "Decode 图化减少 host 下发开销；模式与 batch 形状强相关，换负载需重测。",
        "guards": ["quality", "ttft_p95_ms"],
        "source": "upstream serving guide; profiling notes",
    },
    {
        "key": "prefix_cache",
        "label": "前缀缓存",
        "type": "bool",
        "mutations": [{"kind": "toggle"}],
        "applies": {"hardware": ["A3", "A2", "A5", "950"], "topology": ["1P1D", "PD", "colocated"]},
        "cost": "restart",
        "risk": "medium",
        "expected_pct": 6.0,
        "rationale": "只在有可复用前缀时有效；命中率低时会增加哈希/内存开销，必须带前缀复用负载测。",
        "guards": ["ttft_p95_ms", "quality"],
        "source": "production field notes; connector docs",
    },
    {
        "key": "mtp_depth",
        "label": "MTP 投机深度",
        "type": "enum",
        "values": [1, 3, 5],
        "mutations": [{"kind": "enum"}],
        "applies": {"hardware": ["A3", "A5", "950"]},
        "cost": "restart",
        "risk": "high",
        "expected_pct": 8.0,
        "rationale": "更深的 MTP 可能提高接受长度，也会增加 draft 计算与显存；需同时看 TPOT 与质量。",
        "guards": ["tpot_p95_ms", "quality"],
        "source": "upstream serving guide",
    },
    {
        "key": "hccl_buffsize",
        "label": "HCCL 通信 buffer",
        "type": "enum",
        "values": [1800, 3600],
        "mutations": [{"kind": "enum"}],
        "applies": {"hardware": ["A3", "A5", "950"], "topology": ["1P1D", "PD"]},
        "cost": "restart",
        "risk": "medium",
        "expected_pct": 2.0,
        "rationale": "影响 P/D KV 传输与大 EP 通信；过大可能挤占 HBM。",
        "guards": ["memory_gb", "ttft_p95_ms"],
        "source": "upstream serving guide",
    },
    {
        "key": "dp_tp",
        "label": "DP/TP 拓扑",
        "type": "enum",
        "values": ["keep"],
        "mutations": [{"kind": "none"}],
        "applies": {"hardware": ["A3", "A5", "950"], "topology": ["1P1D", "PD", "colocated"]},
        "cost": "topology",
        "risk": "high",
        "expected_pct": 10.0,
        "rationale": "拓扑变更影响通信域与 KV 布局，实验成本最高；应由候选配置显式给出，不由 planner 自动生成。",
        "guards": ["quality", "memory_gb"],
        "source": "upstream serving and connector docs",
    },
    {
        "key": "kv_layerwise",
        "label": "KV 逐层传输/池化",
        "type": "bool",
        "mutations": [{"kind": "toggle"}],
        "applies": {"hardware": ["A3", "A5", "950"], "topology": ["PD", "1P1D"]},
        "cost": "restart",
        "risk": "high",
        "expected_pct": 5.0,
        "rationale": "只有部分 connector 组合支持 layerwise；需确认版本与 TP 门槛。",
        "guards": ["ttft_p95_ms", "quality"],
        "source": "upstream connector design docs",
    },
    {
        "key": "kv_offload_tier",
        "label": "KV offload 二级 tier",
        "type": "enum",
        "values": ["off", "fs"],
        "mutations": [{"kind": "enum"}],
        "applies": {"hardware": ["A3", "A2", "A5", "950"], "topology": ["PD", "1P1D", "colocated"]},
        "cost": "restart",
        "risk": "medium",
        "expected_pct": 4.0,
        "rationale": "SSD tier 只在冷 miss 请求上有效；warm/HBM 命中时可能无收益甚至回退。",
        "guards": ["ttft_p95_ms", "quality"],
        "source": "production field notes",
    },
    {
        "key": "cpu_bytes_to_use",
        "label": "CPU offload 池大小",
        "type": "int",
        "unit": "bytes",
        "mutations": [{"kind": "scale", "factor": 1.5}, {"kind": "scale", "factor": 0.5}],
        "applies": {"hardware": ["A3", "A2", "A5", "950"], "topology": ["PD", "1P1D", "colocated"]},
        "cost": "restart",
        "risk": "medium",
        "expected_pct": 2.0,
        "rationale": "池越大越能接住长前缀，但会挤占 host 内存并影响换入换出时延。",
        "guards": ["memory_gb"],
        "source": "production field notes",
    },
    {
        "key": "blocks_per_chunk",
        "label": "offload 每块 block 数",
        "type": "enum",
        "values": [1, 2, 4],
        "mutations": [{"kind": "enum"}],
        "applies": {"hardware": ["A3", "A5", "950"], "topology": ["PD", "1P1D", "colocated"]},
        "cost": "restart",
        "risk": "medium",
        "expected_pct": 2.0,
        "rationale": "影响换入换出的批粒度；与 tokens_per_block/tokens_per_hash 对齐是硬前提。",
        "guards": ["ttft_p95_ms", "quality"],
        "source": "production field notes",
    },
    {
        "key": "n_read_threads",
        "label": "SSD tier 读线程数",
        "type": "int",
        "unit": "threads",
        "mutations": [{"kind": "scale", "factor": 2.0}],
        "applies": {"hardware": ["A3", "A5", "950"], "topology": ["PD", "1P1D", "colocated"]},
        "cost": "restart",
        "risk": "low",
        "expected_pct": 3.0,
        "rationale": "冷 miss 时读取并行度影响 TTFT；线程过多会造成 IO 抖动。",
        "guards": ["ttft_p95_ms"],
        "source": "production field notes",
    },
    {
        "key": "n_write_threads",
        "label": "SSD tier 写线程数",
        "type": "int",
        "unit": "threads",
        "mutations": [{"kind": "scale", "factor": 2.0}],
        "applies": {"hardware": ["A3", "A5", "950"], "topology": ["PD", "1P1D", "colocated"]},
        "cost": "restart",
        "risk": "low",
        "expected_pct": 1.0,
        "rationale": "影响池化写入延迟；通常在后台，收益小于读线程。",
        "guards": [],
        "source": "production field notes",
    },
    {
        "key": "quant_scheme",
        "label": "量化方案",
        "type": "enum",
        "values": ["w8a8", "w4a8"],
        "mutations": [{"kind": "enum"}],
        "applies": {"hardware": ["A3", "A2", "A5", "950"]},
        "cost": "restart",
        "risk": "high",
        "expected_pct": 12.0,
        "rationale": "W4A8 可能提升吞吐/降显存，但必须先用精度门禁（MMLU/GSM8K/业务集）再谈速度。",
        "guards": ["quality"],
        "source": "quantization and offload field notes",
    },
]


def _matches_text(actual: str, wanted) -> bool:
    actual = str(actual or "").lower()
    return any(str(w).lower() in actual or actual in str(w).lower() for w in (wanted or []) if str(w))


def knob_applies(knob: dict, scenario) -> bool:
    applies = knob.get("applies") or {}
    hardware = applies.get("hardware") or []
    if hardware:
        actual = " ".join(str(x) for x in (
            scenario.hardware.get("device_model"), scenario.hardware.get("soc")) if x)
        if not _matches_text(actual, hardware):
            return False
    topology = applies.get("topology") or []
    if topology and not _matches_text(scenario.topology, topology):
        return False
    models = applies.get("models") or []
    if models and not _matches_text(scenario.model, models):
        return False
    return True


def get_knob(key: str) -> dict | None:
    for knob in KNOBS:
        if knob["key"] == key:
            return knob
    return None


def default_value(knob: dict):
    if knob["type"] == "bool":
        return False
    if knob["type"] == "int":
        return 0
    if knob["type"] == "enum":
        return knob.get("values", [None])[0]
    return None