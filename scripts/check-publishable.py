#!/usr/bin/env python3
"""Publish-blocking checks that nothing else in the pipeline catches.

Every class here has actually broken a release, and none of them is caught by
`cargo check`, `cargo test`, `cargo clippy` or CI — they only surface from
`cargo publish`, or from a *fresh clone* that lacks files the developing machine
happened to have:

  1. `include_str!`/`include_bytes!` reaching outside its own package. `cargo
     package` ships only files under the crate root, so the published crate
     cannot compile. Invisible locally, because the sibling crate is right
     there. (`cargo package --no-verify` also misses it — verification must run.)
  2. An include target that git ignores. The file is a *compile-time* input, so
     a fresh clone cannot build the target at all — and `clippy --all-targets`
     fails for a reason that does not reproduce on the machine that generated
     the fixture.
  3. A path dependency with no version. `cargo publish` hard-rejects it. Self
     dev-deps (a crate depending on itself to enable test features) are exempt:
     cargo strips those entirely, verified against a scratch crate.
  4. A packaged crate over the crates.io 10 MiB `.crate` cap.
  5. `scripts/publish.sh` TIERS: a crate missing from every tier is silently
     never published, and a dependency sitting in the same or a later tier than
     its dependent fails *partway through* a real publish run. `publish.sh
     --list` validates membership but never ordering.

Exit 0 clean, 1 on any finding.

    python3 scripts/check-publishable.py              # offline, CI-safe
    python3 scripts/check-publishable.py --release    # + the two release gates

`--release` adds the checks that are only meaningful immediately before a
publish run, and that would otherwise fail every day of normal development:

  6. every upstream (non-workspace) `rlx*` pin actually resolves on the
     crates.io sparse index. A dep that was never published makes the whole
     workspace unresolvable from the registry — `cargo metadata` dies outright
     with the local path patch moved aside. Needs network.
  7. the working tree is clean. `cargo publish` refuses uncommitted changes, and
     `--allow-dirty` silently *excludes* untracked files — which is worse: a
     crate whose `mod`-declared sources are untracked publishes and then fails
     to compile for everyone.
"""
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# crates.io rejects a `.crate` over 10 MiB. Overridable only so the check itself
# can be exercised without fabricating a 10 MiB fixture.
CRATE_SIZE_CAP = int(os.environ.get("RLX_CRATE_SIZE_CAP", 10 * 1024 * 1024))
INCLUDE = re.compile(r'include_(?:str|bytes)!\s*\(\s*"([^"]+)"')


def metadata():
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if out.returncode != 0:
        sys.exit(f"cargo metadata failed:\n{out.stderr.strip()}")
    return json.loads(out.stdout)["packages"]


def git_ignored(paths):
    """Subset of `paths` that git ignores (one batched call)."""
    if not paths:
        return set()
    out = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(str(p) for p in paths), capture_output=True, text=True, cwd=ROOT,
    )
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def check_includes(problems):
    """(1) escapes the package, and (2) target is gitignored or missing."""
    targets = []
    for rs in ROOT.glob("crates/*/**/*.rs"):
        if "target" in rs.parts:
            continue
        crate_root = ROOT / rs.relative_to(ROOT).parts[0] / rs.relative_to(ROOT).parts[1]
        try:
            text = rs.read_text(errors="ignore")
        except OSError:
            continue
        for m in INCLUDE.finditer(text):
            resolved = (rs.parent / m.group(1)).resolve()
            rel = rs.relative_to(ROOT)
            try:
                resolved.relative_to(crate_root.resolve())
            except ValueError:
                problems.append(
                    f"{rel}: include_* escapes its package -> {m.group(1)} "
                    f"(cargo package ships only files under the crate root)"
                )
                continue
            if not resolved.exists():
                problems.append(f"{rel}: include_* target does not exist -> {m.group(1)}")
                continue
            targets.append((rel, m.group(1), resolved.relative_to(ROOT)))
    ignored = git_ignored([t[2] for t in targets])
    for rel, raw, target in targets:
        if str(target) in ignored:
            problems.append(
                f"{rel}: include_* target is gitignored -> {raw} "
                f"(compile-time input; a fresh clone cannot build this target)"
            )


def check_path_deps(problems, pkgs):
    for p in sorted(pkgs, key=lambda x: x["name"]):
        if p.get("publish") is not None:
            continue
        for d in p["dependencies"]:
            if not d.get("path") or d.get("req") not in (None, "*"):
                continue
            if d["name"] == p["name"]:
                continue  # self dev-dep: cargo strips it entirely
            problems.append(
                f"{p['name']} -> {d['name']} [{d['kind'] or 'normal'}]: "
                f"path dependency with no version (cargo publish hard-rejects)"
            )


def check_sizes(problems, pkgs):
    for p in pkgs:
        if p.get("publish") is not None:
            continue
        d = Path(p["manifest_path"]).parent
        if not d.is_dir():
            continue
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "."],
            cwd=d, capture_output=True,
        )
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for f in (x for x in out.stdout.decode().split("\0") if x):
                fp = d / f
                if fp.is_file():
                    try:
                        tf.add(fp, arcname=f)
                    except OSError:
                        pass
        if buf.tell() > CRATE_SIZE_CAP:
            problems.append(
                f"{p['name']}: ~{buf.tell() / 1048576:.2f} MiB packaged, over the "
                f"{CRATE_SIZE_CAP / 1048576:.2f} MiB cap"
            )


def check_tiers(problems, pkgs):
    src = (ROOT / "scripts" / "publish.sh").read_text()
    m = re.search(r"TIERS=\((.*?)\n\)", src, re.S)
    if not m:
        problems.append("scripts/publish.sh: could not locate the TIERS array")
        return
    names = {p["name"] for p in pkgs}
    publishable = {p["name"] for p in pkgs if p.get("publish") is None}
    tier_of, seen_in = {}, {}
    for i, line in enumerate(l for l in m.group(1).split("\n") if l.strip()):
        for tok in re.findall(r"[A-Za-z0-9_-]+", line.split("#")[0]):
            if tok in names:
                seen_in.setdefault(tok, []).append(i)
                tier_of[tok] = i
    # A crate listed twice is a silent hazard: the tier it actually publishes in
    # is whichever line came last, which is not what a reader would assume.
    for crate, tiers in sorted(seen_in.items()):
        if len(tiers) > 1:
            problems.append(
                f"publish.sh TIERS: {crate} appears in tiers {tiers} — "
                f"the last one silently wins"
            )
    for crate in sorted(publishable - set(tier_of)):
        problems.append(f"publish.sh TIERS: {crate} is in no tier (silently never published)")
    # Self-deps excluded for the same reason as in check_path_deps: a crate
    # depending on itself to enable test features is not an ordering edge.
    deps = {p["name"]: {d["name"] for d in p["dependencies"]
                        if d["name"] in publishable and d["name"] != p["name"]}
            for p in pkgs if p["name"] in publishable}
    for crate in sorted(publishable & set(tier_of)):
        for dep in sorted(deps[crate]):
            if dep in tier_of and tier_of[dep] >= tier_of[crate]:
                problems.append(
                    f"publish.sh TIERS: {crate} (tier {tier_of[crate]}) depends on "
                    f"{dep} (tier {tier_of[dep]}) — dep must sit in an earlier tier"
                )


def check_upstream_pins(problems):
    """(6) every upstream `rlx*` pin exists on the crates.io index."""
    import urllib.request

    text = (ROOT / "Cargo.toml").read_text()
    table = text[text.index("[workspace.dependencies]"):]
    pins = {}
    for line in table.split("\n"):
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_-]+)\s*=\s*(.+)$", line)
        if not m or not m.group(1).startswith("rlx") or "path" in m.group(2):
            continue
        v = re.search(r'version\s*=\s*"([^"]+)"', m.group(2)) or re.search(r'^"([^"]+)"', m.group(2))
        if v:
            pins[m.group(1)] = v.group(1).lstrip("^=~")
    for name, want in sorted(pins.items()):
        n = name.lower()
        if len(n) <= 2:
            path = f"{len(n)}/{n}"
        elif len(n) == 3:
            path = f"3/{n[0]}/{n}"
        else:
            path = f"{n[:2]}/{n[2:4]}/{n}"
        try:
            with urllib.request.urlopen(f"https://index.crates.io/{path}", timeout=30) as r:
                vers = {json.loads(l)["vers"] for l in r.read().decode().splitlines() if l.strip()}
        except Exception as e:
            problems.append(f"upstream pin {name} {want}: not on the crates.io index ({e})")
            continue
        if want not in vers:
            problems.append(f"upstream pin {name}: {want} not published (index has {len(vers)} versions)")


def check_clean_tree(problems):
    """(7) cargo publish refuses a dirty tree; --allow-dirty drops untracked files."""
    out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT)
    untracked = [l[3:] for l in out.stdout.splitlines() if l.startswith("??")]
    modified = [l[3:] for l in out.stdout.splitlines() if not l.startswith("??")]
    if untracked:
        problems.append(
            f"working tree has {len(untracked)} untracked path(s) — cargo publish "
            f"excludes them, so a crate whose sources are untracked ships broken "
            f"(e.g. {', '.join(untracked[:3])})"
        )
    if modified:
        problems.append(f"working tree has {len(modified)} uncommitted change(s) — cargo publish refuses")


def main():
    release = "--release" in sys.argv[1:]
    pkgs = metadata()
    problems = []
    check_includes(problems)
    check_path_deps(problems, pkgs)
    check_sizes(problems, pkgs)
    check_tiers(problems, pkgs)
    if release:
        check_upstream_pins(problems)
        check_clean_tree(problems)
    if problems:
        print(f"publish preflight: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    n = sum(1 for p in pkgs if p.get("publish") is None)
    scope = "release" if release else "offline"
    print(f"publish preflight OK ({scope}) — {n} publishable crates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
