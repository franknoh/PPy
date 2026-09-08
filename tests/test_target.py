"""`TargetInfo`: one record for the machine a build targets."""

from __future__ import annotations

import pytest

from ppy_compiler.target import TargetError, host_target, parse_target


@pytest.mark.parametrize(
    ("spelled", "canonical", "width", "endianness", "abi", "os", "fmt", "suffix"),
    [
        (
            "aarch64-linux-gnu",
            "aarch64-unknown-linux-gnu",
            64,
            "little",
            "gnu",
            "linux",
            "elf",
            ".so",
        ),
        (
            "x86_64-unknown-linux-gnu",
            "x86_64-unknown-linux-gnu",
            64,
            "little",
            "gnu",
            "linux",
            "elf",
            ".so",
        ),
        (
            "x86_64-linux-musl",
            "x86_64-unknown-linux-musl",
            64,
            "little",
            "musl",
            "linux",
            "elf",
            ".so",
        ),
        (
            "aarch64-apple-darwin",
            "aarch64-apple-darwin",
            64,
            "little",
            "darwin",
            "darwin",
            "macho",
            ".dylib",
        ),
        (
            "arm64-apple-macos",
            "aarch64-apple-macos",
            64,
            "little",
            "darwin",
            "darwin",
            "macho",
            ".dylib",
        ),
        (
            "x86_64-pc-windows-msvc",
            "x86_64-pc-windows-msvc",
            64,
            "little",
            "msvc",
            "windows",
            "coff",
            ".dll",
        ),
        (
            "riscv64-linux-gnu",
            "riscv64-unknown-linux-gnu",
            64,
            "little",
            "gnu",
            "linux",
            "elf",
            ".so",
        ),
        ("s390x-linux-gnu", "s390x-unknown-linux-gnu", 64, "big", "gnu", "linux", "elf", ".so"),
        (
            "armv7-linux-gnueabihf",
            "armv7-unknown-linux-gnueabihf",
            32,
            "little",
            "gnueabihf",
            "linux",
            "elf",
            ".so",
        ),
    ],
)
def test_a_triple_parses_into_the_facts_a_backend_asks_for(
    spelled, canonical, width, endianness, abi, os, fmt, suffix
):
    info = parse_target(spelled)
    assert info.triple == canonical
    assert (info.pointer_width, info.endianness, info.abi, info.os) == (width, endianness, abi, os)
    assert (info.object_format, info.shared_library_suffix) == (fmt, suffix)
    assert info.long_width == (32 if os == "windows" or width == 32 else 64)
    assert info.executable_suffix == (".exe" if os == "windows" else "")
    assert info.extension_suffix == (".pyd" if os == "windows" else ".so")
    assert parse_target(canonical) == parse_target(spelled), "canonical spelling round-trips"


def test_a_triple_that_is_not_one_is_refused_with_the_reason():
    with pytest.raises(TargetError, match="not a target triple"):
        parse_target("x86_64")
    with pytest.raises(TargetError, match="architecture"):
        parse_target("foo-linux-gnu")
    with pytest.raises(TargetError, match="operating system"):
        parse_target("x86_64-plan9")
    with pytest.raises(TargetError, match="ABI"):
        parse_target("x86_64-linux-weird")


def test_the_host_is_a_target_like_any_other():
    host = host_target()
    assert host.is_host and not host.link_flags
    assert host.pointer_width in {32, 64}
    assert host.triple == parse_target(host.triple).triple
    other = parse_target(
        "aarch64-linux-gnu" if host.architecture != "aarch64" else "x86_64-linux-gnu"
    )
    assert not other.is_host
    assert other.link_flags == (f"--target={other.triple}",)
    assert host.describe().startswith(host.triple)


def test_llvm_supplies_the_data_layout_where_it_is_installed():
    pytest.importorskip("llvmlite")
    assert parse_target("aarch64-linux-gnu").data_layout.startswith("e-m:e-")
    assert parse_target("x86_64-pc-windows-msvc").data_layout.startswith("e-m:w-")
    assert parse_target("s390x-linux-gnu").data_layout.startswith("E-")
