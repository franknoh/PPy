"""The optimization report: what the compiler did with each function, as text or JSON (spec 82, 83).

Every remark a pass or a lowering left is filed under a stable category --
`function inlined`, `tensor ops fused`, `bounds guard removed`, ... -- so a
tool reading the JSON can count by category across versions while the text
of a remark stays free to improve.
"""

from __future__ import annotations

import json

__all__ = ["CATEGORIES", "categorize", "optimization_report", "render_report"]

#: (category, the words that file a remark under it), first match wins.
CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("function inlined", ("inlined",)),
    ("columnar ops fused", ("columnar", "column")),
    ("tensor ops fused", ("fused", "fusion")),
    ("Arrow zero-copy view selected", ("zero-copy", "Arrow")),
    ("allocation stack-promoted", ("promoted to values", "stack slot")),
    ("generic specialization reused", ("specialization reused",)),
    ("generic specialization emitted", ("specializ",)),
    ("parallel loop emitted", ("parallel",)),
    ("GPU kernel emitted", ("PTX", "kernel")),
    ("StableHLO region emitted", ("StableHLO", "XLA")),
    ("bounds guard removed", ("bounds",)),
    ("overflow guard proven unnecessary", ("proved", "proven", "overflow")),
    ("sanitizer checks inserted", ("sanitizer",)),
    ("dead code removed", ("global-dce", "removed", "dead")),
    ("coroutine lowered", ("coroutine", "async")),
    ("block merged", ("merged into",)),
)


def categorize(remark: str) -> str:
    """The stable category a remark's wording files it under; `note` when none fits."""
    lowered = remark.lower()
    for category, words in CATEGORIES:
        if any(word.lower() in lowered for word in words):
            return category
    return "note"


def optimization_report(bundle, natives, staged=None, sanitizers=()) -> dict:  # type: ignore[no-untyped-def]
    """Everything the build decided, per module and function."""
    report: dict = {
        "project": bundle.project.root.name,
        "opt_level": bundle.project.config.opt_level,
        "pipeline": bundle.project.config.llvm.pipeline,
        "sanitizers": sorted(sanitizers),
        "modules": {},
        "staged": [],
    }
    for name, native in sorted(natives.items()):
        functions: dict = {}
        for qualname, lowered in sorted(native.functions.items()):
            functions[qualname] = {
                "native": True,
                "signature": str(lowered.signature),
                "bound": lowered.exposed,
                "boundary": lowered.exposure_reason,
                "guards_proved": list(native.proved.get(qualname, ())),
            }
        for qualname, reason in sorted(native.rejected.items()):
            functions[qualname] = {"native": False, "reason": reason}
        remarks = [{"category": categorize(text), "text": text} for text in native.remarks]
        remarks.extend(
            {"category": categorize(text), "text": text, "line": line}
            for line, text in native.fusion_notes
        )
        counts: dict[str, int] = {}
        for remark in remarks:
            counts[remark["category"]] = counts.get(remark["category"], 0) + 1
        report["modules"][name] = {
            "functions": functions,
            "remarks": remarks,
            "categories": dict(sorted(counts.items())),
            "fused_kernels": sorted(native.fused),
            "libraries": list(native.libraries),
        }
    if staged is not None:
        for module_name, entries in sorted(staged.artifacts.items()):
            for function, artifact in sorted(entries.items()):
                report["staged"].append(
                    {
                        "module": module_name,
                        "function": function,
                        "platforms": list(artifact.platforms),
                    }
                )
        for qualname, reason in staged.skipped:
            report["staged"].append({"function": qualname, "skipped": reason})
    return report


def render_report(report: dict) -> str:
    """The report as the terminal shows it."""
    lines = [
        f"optimization report: {report['project']} (O{report['opt_level']}, "
        f"{report['pipeline']} road"
        + (f", sanitizers: {', '.join(report['sanitizers'])}" if report["sanitizers"] else "")
        + ")"
    ]
    for name, module in report["modules"].items():
        lines.append(f"module {name}")
        for qualname, entry in module["functions"].items():
            if entry["native"]:
                how = "bound to Python" if entry["bound"] else "native callers only"
                lines.append(f"  {qualname}: native, {how}")
                if entry["guards_proved"]:
                    lines.append(f"    guards proven unnecessary: {len(entry['guards_proved'])}")
            else:
                lines.append(f"  {qualname}: Python -- {entry['reason']}")
        for category, count in module["categories"].items():
            lines.append(f"  {category}: {count}")
        for remark in module["remarks"]:
            where = f" (line {remark['line']})" if "line" in remark else ""
            lines.append(f"    - {remark['text']}{where}")
        if module["fused_kernels"]:
            lines.append(f"  fused kernels: {', '.join(module['fused_kernels'])}")
    for entry in report["staged"]:
        if "skipped" in entry:
            lines.append(f"staged: {entry['function']} skipped -- {entry['skipped']}")
        else:
            lines.append(
                f"staged: {entry['module']}.{entry['function']} for {', '.join(entry['platforms'])}"
            )
    return "\n".join(lines) + "\n"


def report_json(report: dict) -> str:
    return json.dumps(report, indent=1, sort_keys=True) + "\n"
