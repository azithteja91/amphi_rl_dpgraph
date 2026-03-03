from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/generate_test_report.py <junit_xml> <output_md>")
        return 2

    xml_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    tree = ET.parse(xml_path)
    root = tree.getroot()

    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall("testsuite"))

    rows = []
    total_tests = total_failures = total_errors = total_skipped = 0
    total_time = 0.0

    for suite in suites:
        suite_name = suite.attrib.get("name", "(unnamed suite)")
        tests = int(float(suite.attrib.get("tests", "0")))
        failures = int(float(suite.attrib.get("failures", "0")))
        errors = int(float(suite.attrib.get("errors", "0")))
        skipped = int(float(suite.attrib.get("skipped", "0")))
        time_s = float(suite.attrib.get("time", "0"))

        total_tests += tests
        total_failures += failures
        total_errors += errors
        total_skipped += skipped
        total_time += time_s

        rows.append((suite_name, tests, failures, errors, skipped, time_s))

    passed = total_tests - total_failures - total_errors - total_skipped

    lines = [
        "# Test Results",
        "",
        "## Command",
        "",
        "```bash",
        "pytest -vv --junitxml .artifacts/pytest.xml",
        "```",
        "",
        "## Summary",
        "",
        f"- Total tests: **{total_tests}**",
        f"- Passed: **{passed}**",
        f"- Failed: **{total_failures}**",
        f"- Errors: **{total_errors}**",
        f"- Skipped: **{total_skipped}**",
        f"- Runtime (s): **{round(total_time, 3)}**",
        "",
        "## Per-suite Breakdown",
        "",
        "| Suite | Tests | Failures | Errors | Skipped | Time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for suite_name, tests, failures, errors, skipped, time_s in rows:
        lines.append(
            f"| {suite_name} | {tests} | {failures} | {errors} | {skipped} | {round(time_s, 3)} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
