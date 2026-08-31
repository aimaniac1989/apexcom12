"""Render any Humanoid Olympics event or film a policy attempting it."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess

import mujoco
import numpy as np

from env import EVENTS, OBS_DIM, STATE_DIM, OlympicsSim, instance_score, instance_spec
from env.course import OVERHEAD_GROUP, build_event
from env.sim import _mesh_assets, _scene_xml

OUT = pathlib.Path("renders")


def _lit_model(event: str, frictions, challenge=None) -> mujoco.MjModel:
    """The scored layout with lights added; replay uses this exact helper."""
    layout = build_event(event, challenge)
    xml = _scene_xml(layout)
    # The robot asset does not define a <visual> block, so add one rather than
    # relying on MuJoCo's 640px default offscreen framebuffer.
    xml = xml.replace(
        "<worldbody>",
        '<visual><global offwidth="1600" offheight="900"/></visual>\n  <worldbody>',
        1,
    )
    xml = xml.replace("<worldbody>",
                      '<worldbody>\n'
                      '    <light pos="20 -20 30" dir="-0.3 0.4 -1" directional="true" '
                      'diffuse=".65 .65 .62"/>', 1)
    # Friction is a runtime field in the referee, so set it after compiling here too.
    model = mujoco.MjModel.from_xml_string(xml, _mesh_assets())
    for index, value in enumerate(frictions):
        model.geom(f"course_{index}").friction[0] = value
    return model


def _camera():
    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    mujoco.mjv_defaultCamera(camera)
    mujoco.mjv_defaultOption(option)
    option.geomgroup[OVERHEAD_GROUP] = 1
    return camera, option


def png(pixels, path: pathlib.Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{pixels.shape[1]}x{pixels.shape[0]}", "-i", "-", str(path)],
                   input=pixels.tobytes(), check=True)


def mp4(frame_dir: pathlib.Path, out: pathlib.Path, fps: int = 30) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(frame_dir / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(out)], check=True)


def frames_dir(name: str) -> pathlib.Path:
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()
    return directory


def render_event(sim: OlympicsSim) -> None:
    OUT.mkdir(exist_ok=True)
    model = _lit_model(sim.event, sim.frictions, sim.params.challenge)
    data = mujoco.MjData(model)
    sim.reset()
    data.qpos[:] = sim.data.qpos
    mujoco.mj_forward(model, data)
    camera, option = _camera()
    renderer = mujoco.Renderer(model, height=900, width=1600)
    if sim.event == "sprint_400":
        camera.lookat[:] = [0, 0, 0]
        camera.distance, camera.azimuth, camera.elevation = 190, 125, -55
    else:
        camera.lookat[:] = [max(8.0, sim.layout.finish / 2), 0, 1.0]
        camera.distance, camera.azimuth, camera.elevation = max(18, sim.layout.finish * 0.7), 120, -18
    renderer.update_scene(data, camera=camera, scene_option=option)
    path = OUT / f"{sim.event}.png"
    png(renderer.render(), path)
    print("wrote", path)


def film_run(sim: OlympicsSim, artifact: str, step_cap: int | None) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(artifact, sess_options=options, providers=["CPUExecutionProvider"])
    names = [item.name for item in session.get_inputs()]
    view = _lit_model(sim.event, sim.frictions, sim.params.challenge)
    view_data = mujoco.MjData(view)
    renderer = mujoco.Renderer(view, height=720, width=1280)
    camera, option = _camera()
    frames = frames_dir(f"_{sim.event}_run")
    obs, state, reason, index = sim.reset(), np.zeros((1, STATE_DIM), np.float32), None, 0
    cap = min(sim.max_steps, step_cap) if step_cap else sim.max_steps
    while reason is None:
        action, state = session.run(None, {names[0]: obs.reshape(1, OBS_DIM), names[1]: state})
        result = sim.step(np.asarray(action).ravel(), max_steps=cap)
        obs, reason = result.obs, result.terminal_reason
        if sim.steps % 2 == 0 or reason is not None:
            view_data.qpos[:] = sim.data.qpos
            mujoco.mj_forward(view, view_data)
            camera.lookat[:] = [float(view_data.qpos[0]) + 2, float(view_data.qpos[1]),
                                 float(view_data.qpos[2])]
            camera.distance, camera.azimuth, camera.elevation = 5.5, 128, -12
            renderer.update_scene(view_data, camera=camera, scene_option=option)
            png(renderer.render(), frames / f"f{index:05d}.png")
            index += 1
    score = instance_score(sim.event, reason, sim.progress, sim.steps, cap, sim.metrics)
    out = OUT / f"{sim.event}_run.mp4"
    mp4(frames, out)
    print(f"{sim.event}: {reason}, {sim.distance_m:.2f} m, score {score:.4f}; wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", choices=EVENTS, default="sprint_100")
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run", help="ONNX policy to film")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    sim = OlympicsSim(instance_spec(args.event, args.attempt, args.seed))
    render_event(sim)
    if args.run:
        film_run(sim, args.run, args.max_steps)
