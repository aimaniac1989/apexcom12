"""Export the referee's exact static course geometry as renderer-friendly JSON.

Run from the repository root:

    PYTHONPATH=. python tools/export_course_json.py

The result deliberately uses the same :func:`env.course.build_event` layouts that
MuJoCo receives.  Each surface is an axis-aligned local box described by a world
centre, full dimensions, z-yaw, material kind, and whether it may support the
robot.  The 400 m event therefore contains its real 96 tangent boxes rather
than a visually similar but geometrically different circle.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict

from env.course import (COLOR, EVENTS, EVENT_LABELS, HIGH_JUMP_BARS_M, PLINTH_TOP, TRACK_HALF_W,
                        build_event)


def surface_json(surface) -> dict:
    """One MuJoCo box, using full rather than half extents for renderers."""
    data = asdict(surface)
    data["shape"] = "box"
    data["size_m"] = [2 * data.pop("hx"), 2 * data.pop("hy"), 2 * data.pop("hz")]
    data["center_m"] = [data.pop("x"), data.pop("y"), data.pop("z")]
    data["yaw_rad"] = data.pop("yaw")
    data["rgba"] = [float(value) for value in COLOR[data["kind"]].split()]
    return data


def event_json(event: str) -> dict:
    layout = build_event(event)
    result = {
        "id": event,
        "label": EVENT_LABELS[event],
        "start": {"position_m": [layout.start_x, layout.start_y, PLINTH_TOP],
                  "yaw_rad": layout.start_yaw},
        "finish": {"route_distance_m": layout.finish},
        "surfaces": [surface_json(surface) for surface in layout.surfaces],
    }
    if layout.challenge:
        result["challenge"] = dict(layout.challenge)
    if event == "high_jump":
        result["variants"] = {
            "bar_height_m": list(HIGH_JUMP_BARS_M),
            "bar_center_z_m": [PLINTH_TOP + height for height in HIGH_JUMP_BARS_M],
        }
    return result


def export() -> dict:
    return {
        "schema": "humanoid-olympics.course-layout.v1",
        "units": "metres and radians",
        "coordinate_system": {
            "x": "forward on straight events",
            "y": "runner-left on straight events",
            "z": "up",
            "deck_top_z_m": PLINTH_TOP,
        },
        "lane": {"full_width_m": 2 * TRACK_HALF_W, "half_width_m": TRACK_HALF_W},
        "events": [event_json(event) for event in EVENTS],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/course-layouts.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(export(), indent=2) + "\n")
    print(args.out)
