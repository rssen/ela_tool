"""
Contains utility functions to generate console output with progress indicator.

:author: Robin Senn
"""

from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, TextIO

from blessings import Terminal


class StepName(Enum):
    EXTEND_SYMMETRICALLY = "Extend mirror-symmetrically"
    QUANTIZATION_TABLES = "Calculate and check quantization tables"
    TRANS_RGB_YCBCR = "RGB -> YCbCr"
    TRANS_BLOCK = "Block transformation"
    TRANS_YCBCR_RGB = "YCbCr -> RGB"
    CALC_PSNR = "Calculate PSNR"
    CAL_ELA = "Calculate ELA picture"
    OUT_COMPONENTS = "Write components"
    CALC_ENTROPY = "Calculate entropies"


s = [
    StepName.EXTEND_SYMMETRICALLY,
    StepName.QUANTIZATION_TABLES,
    StepName.TRANS_RGB_YCBCR,
    StepName.TRANS_BLOCK,
    StepName.TRANS_YCBCR_RGB,
    StepName.CALC_PSNR,
]
t = Terminal()


def show_progress_indicator(step_name: StepName):
    def show_progress_decorator(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            show_in_progress(step_name)
            res = func(*args, **kwargs)
            show_completed(step_name)
            return res

        return wrapper

    return show_progress_decorator


def initialize(ela, output):
    print(t.hide_cursor)
    global s
    if ela:
        s.append(StepName.CAL_ELA)
    if output:
        s.extend([StepName.CALC_ENTROPY, StepName.OUT_COMPONENTS])

    print(t.clear())
    a = t.position()
    print(a)
    with t.location():
        print(t.move_x(t.width // 2 - len("*** WELCOME TO THE ELA TOOL ***") // 2) + "*** WELCOME TO THE ELA TOOL ***")
        print(t.move_down + "Step" + t.move_x(40) + "Status")
        for step_name in s:
            print(step_name.value + t.move_x(40) + "[-]")


def show_in_progress(step_name: StepName):
    with t.location(40, s.index(step_name) + 4):
        print("[⟳]")


def show_completed(step_name: StepName):
    with t.location(40, s.index(step_name) + 4):
        print("[🗸]")


def print_results_to_console_and_file(
    image_path: Path,
    psnr: float,
    entropies: Dict[str, float],
    recalculated_quality_value: Optional[float],
    quality: Optional[int],
):
    print(t.move_y(len(s) + 5))
    file = open(f"{image_path.stem}_entropy_psnr_quality.txt", "w+")
    _write_and_print(f"{'component':<10} |  {'entropy'}", file)
    _write_and_print("-" * 21, file)
    for key, value in entropies.items():
        _write_and_print(f"{key:<10} | {value}", file)

    _write_and_print(f"\nPSNR: {psnr:<30}", file)
    if quality is not None:
        _write_and_print(f"Quality: {quality:<30}", file)
    if recalculated_quality_value is not None:
        _write_and_print(f"Recalculated quality value: {recalculated_quality_value:<30}", file)
    file.close()


def _write_and_print(s: str, file: TextIO):
    print(s)
    file.write(f"{s}\n")


def teardown():
    print(t.move(t.height - 1, 0) + t.bold("Program completed"))
