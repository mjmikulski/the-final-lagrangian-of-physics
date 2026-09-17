"""Per-report bookkeeping for the README table, read from git.

For every reports/<dir>/: the APPENDIX-*.md companion files (METHOD section 7) as numbered links in file-name
order (empty cell when there are none), the date the report was first published (committer date of the commit
that added its README.md) and the date of the last change to anything in the directory (appendices, errata,
artifacts). Prints one line per report; `--markdown` prints the three cells in README order. Dates are
YYYY-MM-DD. Remember to paste the regenerated cells into the root README whenever an appendix is merged.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=HERE, text=True).strip()


def report_rows():
    rows = []
    for d in sorted(glob.glob(os.path.join(HERE, "reports", "0*"))):
        rel = os.path.relpath(d, HERE)
        appendices = sorted(os.path.relpath(f, HERE) for f in glob.glob(os.path.join(d, "APPENDIX-*.md")))
        added = git("log", "--diff-filter=A", "--follow", "--format=%cs", "--", f"{rel}/README.md").splitlines()
        first = added[-1] if added else ""
        last = git("log", "-1", "--format=%cs", "--", rel)
        rows.append((os.path.basename(d), appendices, first, last))
    return rows


if __name__ == "__main__":
    md = "--markdown" in sys.argv
    for name, apps, first, last in report_rows():
        links = ", ".join(f"[{i}]({a})" for i, a in enumerate(apps, 1))
        if md:
            print(f"{name[:3]} | {links} | {first} | {last}")
        else:
            print(f"{name}: appendices {len(apps)} {apps}, first published {first}, last updated {last}")
