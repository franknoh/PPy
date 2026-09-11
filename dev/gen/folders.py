"""The example folders the gallery lists, by group: the one source of the count."""

from __future__ import annotations

GROUPS = [
    (
        "Basics",
        [
            "01_basics",
            "02_arbitrary_precision",
            "03_effects_and_contracts",
            "04_classes",
            "08_native_data",
            "10_narrowing",
            "11_numerics",
            "13_value_classes",
            "14_tuples",
            "16_dynamic",
            "17_containers",
            "18_errors",
            "19_strings",
        ],
    ),
    (
        "Native code",
        [
            "12_buffers_and_jit",
            "15_algorithms",
            "28_threads",
            "32_native_memory",
            "33_simd_and_cpu",
            "34_atomics_and_threads",
            "35_parallel_range",
            "36_autodiff",
            "37_aio",
            "40_generics",
            "42_toolbox",
            "43_regex",
        ],
    ),
    ("Accelerators", ["38_cuda", "44_tile", "39_xla"]),
    (
        "Libraries",
        [
            "05_numpy",
            "06_pydantic",
            "07_parallel",
            "09_torch",
            "21_training_torch",
            "22_training_jax",
            "25_jax_export",
            "27_uvicorn",
            "29_flax",
            "31_torchrun",
            "41_columnar",
        ],
    ),
    (
        "Conversion and projects",
        ["20_inventory", "23_inference", "24_interop", "26_project", "30_migrate"],
    ),
]
FOLDERS = [folder for _, folders in GROUPS for folder in folders]
