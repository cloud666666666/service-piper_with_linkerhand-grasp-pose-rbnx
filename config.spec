# Public deploy config for robonix.service.piper_with_linkerhand.grasp_pose.
# Values below are the ones this deploy uses; the type/unit/constraint note
# above each key is the contract.
config:
  # string, default: flat; accepted values: flat, vertical.
  # `flat` is this robot's path: flat-palm enveloping grasp, where the wrist
  # keeps its taught orientation and is only rotated about world z to face the
  # object. `vertical` is the upstream top-down path (bbox centre -> homography
  # -> yaw from the orientation label, else from the bbox long edge).
  # Must match the IK service's grasp_mode.
  grasp_mode: flat

  # ── hand-eye ────────────────────────────────────────────────────────────
  # 3x3 matrix (list of 3 rows). Pixel -> arm-plane homography. Supplying it
  # inline keeps the deploy self-contained: the .npz the calibration produces
  # lives outside this repo (upstream .gitignore excludes it).
  homography_matrix:
    - [-1.3703248148934225e-04, -1.3381221012053474e-04, 2.4087601472097991e-01]
    - [1.6924070873515303e-04, -1.7109859294771277e-04, -5.0571730014946012e-02]
    - [-3.4753264175198473e-04, -9.4451286962524443e-04, 1.0000000000000000e+00]

  # string, optional. Load the homography from a calibration .npz instead of
  # inlining it. The .npz also carries heading_offset_deg / L_m, so a re-fit
  # cannot be paired with a stale offset. (`homography_path` is an accepted
  # alias; `hand_eye_calibration_file` is the pre-rename name.)
  # homography_file: /path/to/2d_homography_flat.npz

  # float, degrees, default: 0.0. Taught corridor offset added to the palm's
  # bearing: heading = azimuth(palm) + this. Must match the flat_grasp skill's
  # place_heading_offset_deg.
  heading_offset_deg: 4.2

  # float, metres, default: 0.0. Flange = palm + L * u(heading). 0 means the
  # flange and the palm are treated as coincident.
  arm_length_m: 0.0

  # float, metres, REQUIRED in flat mode — deliberately has no default. The
  # ABSOLUTE grasp height. NOT default_desktop_height: the vendor config states
  # that value does not match the flat-grasp scene, and using it grasps ~2.5 cm
  # high and misses the object.
  flat_grasp_height_m: 0.11

  # float, metres, REQUIRED in vertical mode. Tabletop height used by the
  # vertical path only. Absent here because grasp_mode is flat.
  # default_desktop_height: 0.135

  # float, metres, default: 0.01 (this deploy: 0.0). Extra outward shift along
  # the heading so the palm does not end up jammed against the object. The sign
  # convention differs from the vertical path — do not unify them.
  catch_offset: 0.0

  # integer, pixels, default: 1280. Image width the homography was solved for;
  # used to clamp the short-edge width projection. MUST match the camera
  # profile: the calibration spans u 510..1109, and grasp_pose has no way to
  # notice a narrower stream — it would just send the arm somewhere wrong.
  frame_width: 1280

  # string, default: arm/base_link. Frame the pose is reported in.
  output_frame: arm/base_link

  # ── vertical-path extras (inert in flat mode) ───────────────────────────
  # float, metres, default: 0.0 each. Global correction added to the projected
  # x / y.
  bias_x: 0.0
  bias_y: 0.0

  # float, degrees, default: 0.0. Rotation applied to the detected box.
  box_rotation_deg: 0.0

  # float, metres, default: 0.04. Gripper opening echoed back to the caller;
  # this robot has no gripper (see the IK service's config.spec).
  gripper_width_default: 0.04
