#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Roboarm-style grasp-pose estimator.

This service owns ``robonix/service/perception/grasp_pose/*`` and exposes one
runtime RPC: ``grasp_request``. It does not publish legacy ROS topics and does
not host the old ``/graspnet/grasp_request`` service; downstream execution is
handled explicitly by ``pick_skill -> roboarm_ik``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Any

from robonix_api import Service, Ok, Err  # noqa: E402

logging.basicConfig(
    level=os.environ.get("GRASP_POSE_LOG_LEVEL", "INFO"),
    format="[grasp_pose] %(message)s",
)
log = logging.getLogger("grasp_pose")

grasp_pose = Service(
    id=os.environ.get("ROBONIX_CAPABILITY_ID", "grasp_pose"),
    namespace="robonix/service/perception/grasp_pose",
)

_DEFAULT_CATCH_OFFSET = 0.01
_DEFAULT_BOX_ROTATION_DEG = 0.0
_DEFAULT_BIAS_X = 0.0
_DEFAULT_BIAS_Y = 0.0
_DEFAULT_BASE_FRAME = "arm/base_link"

_state_lock = threading.Lock()
_initialized = False
_resolved_cfg: dict[str, Any] = {}
_homography_matrix = None
# Flat-grasp calibration extras. The joint palm+arm-length model in
# arm/flat_handeye.py fits `palm = flange - L*u(azimuth(palm) + phi0)`; `H` alone
# is only the pixel -> palm-plane half of it. `L` is 0 for the current
# calibration (see _compute_flat_grasp), but keep it explicit rather than
# assuming — a re-fit with more points can legitimately produce L > 0.
_heading_offset_deg: float = 0.0
_arm_length_m: float = 0.0


def _flat_base_rotation(flat_euler_deg_zyx: list[float]):
    """The taught flat-grasp base orientation as a scipy Rotation.

    `linker_hand_flat_euler_deg_zyx` is [RZ, RY, RX] in degrees (roboarm's
    config.yaml.example: [-142.249, 81.015, 149.475], taught on the real arm).

    NOTE the LOWERCASE "zyx". scipy reads uppercase as *intrinsic* rotations
    (Rz @ Ry @ Rx) and lowercase as *extrinsic* (Rx @ Ry @ Rz); the two differ
    by 73.5 deg for these angles, and only the lowercase form matches what
    roboarm does (arm/piper_ctrl_by_sdk_flat_hand.py:move_to_flat uses
    `R.from_euler("zyx", self.flat_euler_deg_zyx, degrees=True)`). Verified
    against the reference's own published benchmark — see the module docstring
    of service-piper_with_linkerhand-ik-rbnx/roboarm_ik/solver.py:solve_flat.
    """
    from scipy.spatial.transform import Rotation as R

    return R.from_euler("zyx", [float(v) for v in flat_euler_deg_zyx], degrees=True)


def _vertical_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    """Quaternion for roboarm's vertical-down grasp convention."""
    half_yaw = float(yaw_rad) * 0.5
    return float(math.sin(half_yaw)), float(math.cos(half_yaw)), 0.0, 0.0


def _load_homography(cfg: dict[str, Any]):
    """Load the pixel -> arm-plane homography plus the flat-model extras.

    Returns ``(H, heading_offset_deg, L_m)``.

    Three accepted sources, in priority order:
      * ``homography_matrix`` inline (9 numbers) — self-contained in the
        manifest, which is what this deploy uses. The calibration .npz lives
        outside the repo (it is gitignored upstream), so inlining keeps the
        container from depending on a host path that may not be mounted.
      * ``homography_file`` — a ``.npz`` from arm/calibrate_arm_hand.py, or a
        plain ``.npy`` 3x3 (the older vertical-grasp format).
      * ``hand_eye_calibration_file`` / ``homography_path`` — legacy key names,
        kept so the reference deploy's manifests still load.
    """
    import numpy as np

    heading_offset_deg = cfg.get("heading_offset_deg")
    arm_length_m = cfg.get("arm_length_m")

    inline = cfg.get("homography_matrix")
    if inline is not None:
        mat = np.asarray(inline, dtype=np.float64)
        # Flat-model extras have no home in a bare 3x3, so they must come from
        # config alongside it.
        return (
            _check_homography(mat),
            float(heading_offset_deg or 0.0),
            float(arm_length_m or 0.0),
        )

    raw_path = (
        cfg.get("homography_file")
        or cfg.get("hand_eye_calibration_file")
        or cfg.get("homography_path")
    )
    if not raw_path:
        raise ValueError(
            "missing homography config: set homography_file to a .npz/.npy "
            "calibration, or provide homography_matrix inline"
        )
    path = os.path.expandvars(os.path.expanduser(str(raw_path)))
    if not os.path.isabs(path):
        pkg_root = os.environ.get(
            "RBNX_PACKAGE_ROOT",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        )
        path = os.path.join(pkg_root, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"homography_file does not exist: {path}")

    if path.endswith(".npz"):
        # arm/flat_handeye.py's save_mapping() output: H + the joint-model
        # scalars + the raw fit statistics. Taking the scalars from the file
        # rather than from config means a re-calibration cannot silently pair a
        # new H with a stale heading_offset_deg.
        with np.load(path, allow_pickle=False) as z:
            if "H" not in z:
                raise ValueError(f"{path}: expected key 'H' in the .npz")
            mat = np.asarray(z["H"], dtype=np.float64)
            if heading_offset_deg is None and "heading_offset_deg" in z:
                heading_offset_deg = float(z["heading_offset_deg"])
            if arm_length_m is None and "L_m" in z:
                arm_length_m = float(z["L_m"])
            # Surface the fit quality, because a degenerate fit is silent:
            # flat_handeye.py warns that with n < RANSAC_MIN_POINTS (15) the
            # RANSAC path picks a minimal set, fits it exactly, dumps the rest
            # as outliers and drives L to a search boundary — the reported RMS
            # then looks good while the mapping is unusable.
            n = int(z["n"]) if "n" in z else -1
            rms = float(z["rms_m"]) if "rms_m" in z else float("nan")
            log.info(
                "loaded flat calibration %s: n=%d rms=%.4f m L=%.4f m "
                "heading_offset=%.2f deg",
                path, n, rms, float(arm_length_m or 0.0),
                float(heading_offset_deg or 0.0),
            )
            if 0 < n < 8:
                log.warning(
                    "calibration has only %d points (flat_handeye.py recommends "
                    ">= 8). The fit is likely degenerate — re-run "
                    "arm/calibrate_arm_hand.py before trusting grasps.", n,
                )
            if arm_length_m is not None and (
                abs(float(arm_length_m)) < 1e-9
                or abs(float(arm_length_m) - 0.25) < 5e-4
            ):
                log.warning(
                    "arm length L=%.4f sits on the search boundary (0 or 0.25), "
                    "which flat_handeye.py treats as unidentifiable — the "
                    "palm/flange offset is not actually being modelled.",
                    float(arm_length_m),
                )
        return _check_homography(mat), float(heading_offset_deg or 0.0), float(
            arm_length_m or 0.0
        )

    mat = np.load(path)
    return (
        _check_homography(mat),
        float(heading_offset_deg or 0.0),
        float(arm_length_m or 0.0),
    )


def _check_homography(mat):
    """Validate a candidate 3x3 homography."""
    import numpy as np

    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"homography matrix must be shape (3, 3), got {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("homography matrix contains non-finite values")
    return mat.astype(np.float64)


def _load_homography_matrix(cfg: dict[str, Any]):
    """Back-compat shim: the vertical path only ever wanted the 3x3."""
    return _load_homography(cfg)[0]


def _pixel_to_arm_xy(u: float, v: float) -> tuple[float, float]:
    """Project an RGB pixel center to arm-plane XY."""
    import numpy as np

    if _homography_matrix is None:
        raise ValueError("homography matrix is not loaded")
    pixel_coords = np.array([[float(u)], [float(v)], [1.0]], dtype=np.float64)
    world_coords = _homography_matrix @ pixel_coords
    denom = float(world_coords[2, 0])
    if abs(denom) < 1e-12:
        raise ValueError("homography projection has near-zero scale")
    world_coords /= denom
    return float(world_coords[0, 0]), float(world_coords[1, 0])


def _gripper_angle_by_longer(
    u: float, v: float, w: float, h: float, angle_deg: float
) -> float:
    """roboarm Arm.gripper_angle_by_longer() without requiring cv2."""
    import numpy as np

    theta = math.radians(float(angle_deg))
    c, s = math.cos(theta), math.sin(theta)
    half_w, half_h = float(w) / 2.0, float(h) / 2.0
    local = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float64,
    )
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    box_points = local @ rot.T + np.array([float(u), float(v)], dtype=np.float64)

    edge_01 = np.linalg.norm(box_points[0] - box_points[1])
    edge_12 = np.linalg.norm(box_points[1] - box_points[2])
    if edge_01 > edge_12:
        long_edge_points = (
            [box_points[0], box_points[1]]
            if box_points[0][0] < box_points[1][0]
            else [box_points[1], box_points[0]]
        )
    else:
        long_edge_points = (
            [box_points[1], box_points[2]]
            if box_points[1][0] < box_points[2][0]
            else [box_points[2], box_points[1]]
        )

    gripper_rot_rad = math.pi / 2 + math.atan2(
        float(long_edge_points[1][1] - long_edge_points[0][1]),
        float(long_edge_points[1][0] - long_edge_points[0][0]),
    )
    if gripper_rot_rad > math.pi / 2:
        gripper_rot_rad -= math.pi
    return float(gripper_rot_rad)


def _apply_xy_bias(x: float, y: float, cfg: dict[str, Any]) -> tuple[float, float]:
    bias_x = float(cfg.get("bias_x", _DEFAULT_BIAS_X))
    bias_y = float(cfg.get("bias_y", _DEFAULT_BIAS_Y))
    return x + bias_x, y + bias_y


def _failure(message: str, frame_id: str) -> dict[str, Any]:
    return {
        "success": False,
        "message": message,
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "frame_id": frame_id,
        "gripper_width": 0.0,
        "score": 0.0,
    }


def _compute_flat_grasp(
    *,
    bbox_2d: list[float],
    cfg: dict[str, Any],
    object_name: str = "",
) -> dict[str, Any]:
    """Compute a flat-palm enveloping grasp pose from a detector bbox.

    Geometry, mirroring roboarm's classification/catch_with_linker_hand_flat.py
    + arm/flat_handeye.py:

      1. bbox center pixel -> PALM point through the homography.
      2. ``heading = azimuth(palm) + phi0`` — the palm's own bearing in the arm
         base frame, plus the taught corridor offset. This is what makes the
         flat grasp different from the vertical one: the wrist is held in the
         taught configuration and only *rotated about world z* to face the
         object, rather than being re-aimed per-object from the bbox shape.
         roboarm's script says so explicitly ("不再用物体长边角").
      3. ``flange = palm + L * u(heading)`` — from flat_handeye.flange_from_palm
         with PALM_MODEL_SIGN = -1. Note the arm-length term uses
         ``azimuth(palm) + phi0``, i.e. the *same* angle as the heading, which
         is why it collapses to one unit vector here.
      4. Shift by ``catch_offset`` along the heading, outward, so the palm does
         not end up jammed against the object
         (catch_with_linker_hand_flat.py: "沿走廊朝向再外偏一份 catch_offset").
         The sign convention differs from the vertical path, which uses
         sin(-yaw) — do not unify them.
      5. z is ``flat_grasp_height_m``, an ABSOLUTE height. NOT
         default_desktop_height: roboarm's config states that value does not
         match the flat-grasp scene, and swapping them grasps ~2.5 cm high.

    The returned quaternion is the full target orientation
    ``Rz(heading) @ R_taught``, which is exactly what move_to_flat(rot_rad=
    heading) builds on the robot. Pinning the wrist joints stays an IK-side
    concern (roboarm's linker_hand_pin_joints), not a pose property.
    """
    import numpy as np

    output_frame = str(cfg.get("output_frame", _DEFAULT_BASE_FRAME))
    if "flat_grasp_height_m" not in cfg:
        return _failure(
            "missing required flat-grasp config: flat_grasp_height_m "
            "(absolute grasp z in meters; do NOT substitute "
            "default_desktop_height, which does not match the flat-grasp scene)",
            output_frame,
        )

    if not bbox_2d or len(bbox_2d) not in (4, 5):
        return _failure(f"bbox_2d must be length 4 or 5, got {bbox_2d!r}",
                        output_frame)

    grasp_z = float(cfg["flat_grasp_height_m"])
    catch_offset = float(cfg.get("catch_offset", 0.0))
    flat_euler = cfg.get("flat_euler_deg_zyx")
    if not flat_euler or len(flat_euler) != 3:
        return _failure(
            "missing required flat-grasp config: flat_euler_deg_zyx "
            "([RZ, RY, RX] degrees — the taught base orientation the heading "
            "is applied on top of)",
            output_frame,
        )

    x_min, y_min, x_max, y_max = (float(value) for value in bbox_2d[:4])
    u = 0.5 * (x_min + x_max)
    v_pix = 0.5 * (y_min + y_max)
    bbox_w = abs(x_max - x_min)
    bbox_h = abs(y_max - y_min)
    if bbox_w <= 0.0 or bbox_h <= 0.0:
        return _failure(f"bbox has non-positive size: {bbox_2d!r}", output_frame)

    try:
        palm_x, palm_y = _pixel_to_arm_xy(u, v_pix)
    except ValueError as e:
        return _failure(f"pixel2pos failed: {e}", output_frame)

    heading_rad = math.atan2(palm_y, palm_x) + math.radians(_heading_offset_deg)
    flange_x = palm_x + _arm_length_m * math.cos(heading_rad)
    flange_y = palm_y + _arm_length_m * math.sin(heading_rad)
    flange_x, flange_y = _apply_xy_bias(flange_x, flange_y, cfg)
    grasp_x = flange_x + catch_offset * math.cos(heading_rad)
    grasp_y = flange_y + catch_offset * math.sin(heading_rad)

    try:
        rot = _flat_base_rotation(flat_euler)
        target_rot = _rotation_about_z(heading_rad) * rot
        qx, qy, qz, qw = (float(v) for v in target_rot.as_quat())
    except Exception as e:  # noqa: BLE001
        return _failure(f"bad flat_euler_deg_zyx {flat_euler!r}: {e}", output_frame)

    # Object width, used by the hand's grasp_by_size() as the enveloping
    # opening. roboarm measures it by projecting the bbox SHORT edge onto the
    # palm plane and taking the distance — pixel distances are not metric, so
    # this has to go through the homography.
    object_width_m = 0.0
    frame_w = float(cfg.get("frame_width", 1280))
    if frame_w > 0:
        short_edge_px = min(bbox_w, bbox_h)
        try:
            side_x, side_y = _pixel_to_arm_xy(min(u + short_edge_px, frame_w - 1), v_pix)
            object_width_m = float(math.hypot(side_x - palm_x, side_y - palm_y))
        except ValueError:
            pass

    log.info(
        "flat grasp: object=%r uv=(%.1f,%.1f) bbox=(%.1fx%.1f) "
        "palm_xy=(%.3f,%.3f) heading=%.4f rad (%.2f deg) catch_offset=%.4f "
        "-> grasp=(x=%.3f, y=%.3f, z=%.3f) width=%.3f m",
        object_name, u, v_pix, bbox_w, bbox_h, palm_x, palm_y,
        heading_rad, math.degrees(heading_rad), catch_offset,
        grasp_x, grasp_y, grasp_z, object_width_m,
    )

    return {
        "success": True,
        "message": (
            f"ok (object={object_name!r}, u,v=({u:.1f},{v_pix:.1f}), "
            f"palm=({palm_x:.3f},{palm_y:.3f}), "
            f"grasp=({grasp_x:.3f},{grasp_y:.3f},{grasp_z:.3f}), "
            f"heading={heading_rad:.4f} rad)"
        ),
        "pose": {
            "position": {
                "x": float(grasp_x),
                "y": float(grasp_y),
                "z": float(grasp_z),
            },
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
        "frame_id": output_frame,
        # For the flat grasp this is the MEASURED object width, not a commanded
        # opening: the hand closes with a fixed fist pose and the width is what
        # grasp_by_size() would interpolate against. Kept in this field because
        # the IDL has no better home and downstream already reads it.
        "gripper_width": float(object_width_m),
        "score": 0.8,
    }


def _rotation_about_z(angle_rad: float):
    """Rotation of `angle_rad` about the world z axis (scipy Rotation)."""
    from scipy.spatial.transform import Rotation as R

    return R.from_euler("Z", float(angle_rad))


def _compute_grasp(
    *,
    bbox_2d: list[float],
    cfg: dict[str, Any],
    object_name: str = "",
) -> dict[str, Any]:
    """Dispatch to the configured grasp model."""
    mode = str(cfg.get("grasp_mode", "flat")).lower()
    if mode == "flat":
        return _compute_flat_grasp(
            bbox_2d=bbox_2d, cfg=cfg, object_name=object_name
        )
    if mode == "vertical":
        return _compute_vertical_grasp(
            bbox_2d=bbox_2d, cfg=cfg, object_name=object_name
        )
    return _failure(
        f"unknown grasp_mode {mode!r} (expected 'flat' or 'vertical')",
        str(cfg.get("output_frame", _DEFAULT_BASE_FRAME)),
    )


def _compute_vertical_grasp(
    *,
    bbox_2d: list[float],
    cfg: dict[str, Any],
    object_name: str = "",
) -> dict[str, Any]:
    """Vertical grasp: a downward gripper pose, yaw from the bbox long edge.

    Kept from the reference deploy (/home/czn/wfw/robot-agilex-piper) unchanged
    apart from being reachable only via ``grasp_mode: vertical``. This robot
    has a dexterous hand and grasps flat, so the flat path is the default —
    but the vertical algorithm is the one proven on this same arm with a
    gripper, and keeping it makes the two comparable.
    """
    output_frame = str(cfg.get("output_frame", _DEFAULT_BASE_FRAME))
    if "default_desktop_height" not in cfg:
        return _failure(
            "missing required roboarm config: default_desktop_height",
            output_frame,
        )

    if not bbox_2d or len(bbox_2d) not in (4, 5):
        return _failure(f"bbox_2d must be length 4 or 5, got {bbox_2d!r}",
                        output_frame)

    desktop_height = float(cfg["default_desktop_height"])
    catch_offset = float(cfg.get("catch_offset", _DEFAULT_CATCH_OFFSET))
    width = float(cfg.get("gripper_width_default", 0.04))
    box_rotation_deg = float(
        cfg.get("box_rotation_deg", _DEFAULT_BOX_ROTATION_DEG)
    )

    x_min, y_min, x_max, y_max = (float(value) for value in bbox_2d[:4])
    if len(bbox_2d) == 5:
        box_rotation_deg = float(bbox_2d[4])
    u = 0.5 * (x_min + x_max)
    v_pix = 0.5 * (y_min + y_max)
    bbox_w = abs(x_max - x_min)
    bbox_h = abs(y_max - y_min)
    if bbox_w <= 0.0 or bbox_h <= 0.0:
        return _failure(f"bbox has non-positive size: {bbox_2d!r}",
                        output_frame)

    try:
        raw_x, raw_y = _pixel_to_arm_xy(u, v_pix)
        target_x, target_y = _apply_xy_bias(raw_x, raw_y, cfg)
    except ValueError as e:
        return _failure(f"pixel2pos failed: {e}", output_frame)

    yaw_rad = _gripper_angle_by_longer(
        u, v_pix, bbox_w, bbox_h, box_rotation_deg
    )
    catch_dx = catch_offset * math.cos(yaw_rad)
    catch_dy = catch_offset * math.sin(-yaw_rad)
    grasp_x = target_x + catch_dx
    grasp_y = target_y + catch_dy
    qx, qy, qz, qw = _vertical_quaternion(yaw_rad)

    bias_x = float(cfg.get("bias_x", _DEFAULT_BIAS_X))
    bias_y = float(cfg.get("bias_y", _DEFAULT_BIAS_Y))
    log.info(
        "roboarm grasp: object=%r uv=(%.1f,%.1f) bbox=(%.1fx%.1f, rot=%.1f) "
        "raw_xy=(x=%.3f, y=%.3f) bias=(%.3f, %.3f) biased_xy=(x=%.3f, y=%.3f) "
        "catch_offset=(dx=%.3f, dy=%.3f) -> grasp=(x=%.3f, y=%.3f, z=%.3f) yaw=%.3f",
        object_name, u, v_pix, bbox_w, bbox_h, box_rotation_deg,
        raw_x, raw_y, bias_x, bias_y, target_x, target_y,
        catch_dx, catch_dy, grasp_x, grasp_y, desktop_height, yaw_rad)

    return {
        "success": True,
        "message": (
            f"ok (object={object_name!r}, u,v=({u:.1f},{v_pix:.1f}), "
            f"arm_xy=({grasp_x:.3f},{grasp_y:.3f}), yaw={yaw_rad:.3f})"
        ),
        "pose": {
            "position": {
                "x": float(grasp_x),
                "y": float(grasp_y),
                "z": float(desktop_height),
            },
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
        "frame_id": output_frame,
        "gripper_width": width,
        "score": 0.8,
    }


def _serve_grasp_request(*, object_name: str, bbox_2d: list[float]) -> dict[str, Any]:
    if not bbox_2d:
        return _failure(
            "bbox_2d is required; pick_skill should call object_detect first",
            str(_resolved_cfg.get("output_frame", _DEFAULT_BASE_FRAME)),
        )
    return _compute_grasp(
        bbox_2d=bbox_2d,
        cfg=_resolved_cfg,
        object_name=object_name,
    )


@grasp_pose.on_init
def init(cfg):
    """Driver(CMD_INIT): parse config and load the hand-eye homography."""
    global _initialized, _resolved_cfg, _homography_matrix
    global _heading_offset_deg, _arm_length_m
    with _state_lock:
        if _initialized:
            return Ok()

    cfg = cfg or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg) if cfg else {}
        except json.JSONDecodeError as e:
            return Err(f"bad config_json: {e}")

    mode = str(cfg.get("grasp_mode", "flat")).lower()
    if mode not in ("flat", "vertical"):
        return Err(f"unknown grasp_mode {mode!r} (expected 'flat' or 'vertical')")

    if mode == "flat":
        # flat_grasp_height_m is the ABSOLUTE grasp height. It is deliberately
        # not default_desktop_height — see the _compute_flat_grasp docstring.
        if "flat_grasp_height_m" not in cfg:
            return Err(
                "missing required flat-grasp config: flat_grasp_height_m "
                "(absolute grasp z in meters, roboarm's "
                "linker_hand_flat_grasp_height_m, default 0.11)"
            )
        try:
            float(cfg["flat_grasp_height_m"])
        except Exception as e:  # noqa: BLE001
            return Err(f"bad flat_grasp_height_m: {e}")
        flat_euler = cfg.get("flat_euler_deg_zyx")
        if not flat_euler or len(flat_euler) != 3:
            return Err(
                "missing required flat-grasp config: flat_euler_deg_zyx "
                "(roboarm's linker_hand_flat_euler_deg_zyx, [RZ, RY, RX] deg)"
            )
        try:
            _flat_base_rotation(flat_euler)
        except Exception as e:  # noqa: BLE001
            return Err(f"bad flat_euler_deg_zyx {flat_euler!r}: {e}")
    else:
        if "default_desktop_height" not in cfg:
            return Err(
                "missing required roboarm config: default_desktop_height "
                "(meters, arm base frame z for the grasp height)"
            )
        try:
            float(cfg["default_desktop_height"])
        except Exception as e:  # noqa: BLE001
            return Err(f"bad default_desktop_height: {e}")

    try:
        _homography_matrix, heading_offset_deg, arm_length_m = _load_homography(cfg)
        _heading_offset_deg = heading_offset_deg
        _arm_length_m = arm_length_m
    except Exception as e:  # noqa: BLE001
        return Err(f"bad roboarm homography config: {e}")

    _resolved_cfg = cfg
    with _state_lock:
        _initialized = True
    log.info(
        "init complete: mode=%s, gRPC grasp_request live (cfg keys=%d)",
        mode, len(cfg),
    )
    return Ok()


@grasp_pose.on_deactivate
def deactivate():
    global _initialized
    with _state_lock:
        _initialized = False
    log.info("CMD_DEACTIVATE ok")
    return Ok()


import grasp_pb2  # noqa: E402  pylint: disable=wrong-import-position
import geometry_msgs_pb2  # noqa: E402
import std_msgs_pb2  # noqa: E402
import builtin_interfaces_pb2  # noqa: E402


@grasp_pose.grpc("robonix/service/perception/grasp_pose/grasp_request")
def grasp_request(req: grasp_pb2.GraspRequest_Request) -> grasp_pb2.GraspRequest_Response:
    """Compute a grasp pose from a caller-supplied RGB bbox."""
    result = _serve_grasp_request(
        object_name=req.object_name,
        bbox_2d=list(req.bbox_2d) if req.bbox_2d else [],
    )
    p = result["pose"]
    pose_stamped = geometry_msgs_pb2.PoseStamped(
        header=std_msgs_pb2.Header(
            stamp=builtin_interfaces_pb2.Time(sec=0, nanosec=0),
            frame_id=result["frame_id"],
        ),
        pose=geometry_msgs_pb2.Pose(
            position=geometry_msgs_pb2.Point(
                x=float(p["position"]["x"]),
                y=float(p["position"]["y"]),
                z=float(p["position"]["z"])),
            orientation=geometry_msgs_pb2.Quaternion(
                x=float(p["orientation"]["x"]),
                y=float(p["orientation"]["y"]),
                z=float(p["orientation"]["z"]),
                w=float(p["orientation"]["w"])),
        ),
    )
    return grasp_pb2.GraspRequest_Response(
        grasp_pose=pose_stamped,
        gripper_width=float(result["gripper_width"]),
        score=float(result["score"]),
        success=bool(result["success"]),
        message=str(result["message"]),
    )


def main() -> int:
    grasp_pose.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
