"""Rendering utilities (mesh overlay, video, etc.).

This module requires optional dependencies: pyrender, open3d, utils3d, imageio.
All imports are guarded so the rest of the package works without them.
"""
import os
import sys
import contextlib

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('FILAMENT_LOG_LEVEL', 'error')
os.environ.setdefault('EGL_LOG_LEVEL', 'fatal')
if os.environ.get('PYOPENGL_PLATFORM') == 'osmesa':
    os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '4.1')
    os.environ.setdefault('MESA_GLSL_VERSION_OVERRIDE', '410')

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R
import trimesh

try:
    import utils3d
    import open3d as o3d
    import pyrender
    from ..renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
    from ..representations import Octree, Gaussian, MeshExtractResult
    from ..modules import sparse as sp
    from .random_utils import sphere_hammersley_sequence
    from .pose_utils import parse_pose_output
    _HAS_VIS_DEPS = True
except ImportError:
    _HAS_VIS_DEPS = False

# Suppress Open3D and Filament logs
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr at the OS level."""
    # Save the actual stdout/stderr file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    # Duplicate the file descriptors so we can restore them later
    with os.fdopen(os.dup(stdout_fd), 'w') as old_stdout, \
         os.fdopen(os.dup(stderr_fd), 'w') as old_stderr:
        
        # Open devnull
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        
        try:
            # Redirect stdout and stderr to devnull at the OS level
            os.dup2(devnull_fd, stdout_fd)
            os.dup2(devnull_fd, stderr_fd)
            yield
        finally:
            # Restore stdout and stderr
            os.dup2(old_stdout.fileno(), stdout_fd)
            os.dup2(old_stderr.fileno(), stderr_fd)
            os.close(devnull_fd)


def coord_transform():
    """Transform from OpenCV camera coordinates to OpenGL camera coordinates."""
    t = np.eye(4)
    t[:3, :3] = np.array([[1, 0, 0],
                     [0, -1, 0],
                     [0, 0, -1]])
    return t


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def get_renderer(sample, **kwargs):
    if isinstance(sample, Octree):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 1)
        renderer.rendering_options.far = kwargs.get('far', 100)
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    renderer = get_renderer(sample, **options)
    rets = {}
    return_types = kwargs.get('return_types', None)
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if isinstance(sample, MeshExtractResult):
            # Support custom return types for mesh rendering
            mesh_return_types = return_types if return_types is not None else ['normal']
            res = renderer.render(sample, extr, intr, return_types=mesh_return_types)
            for ret_type in mesh_return_types:
                if ret_type not in rets:
                    rets[ret_type] = []
                if ret_type in res:
                    rets[ret_type].append(np.clip(res[ret_type].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        else:
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_mesh_overlay(mesh, pred_pose_tensor, cam2ncam, intrinsics, original_image_pil, image_size=None, skip_glb_rotation=False, pose_representation=None):
    """Render generated mesh overlaid on original image from camera view.
    
    Args:
        mesh: Trimesh object with vertices and faces
        pred_pose_tensor: Pose tensor (auto-detects format: 8D quaternion, 10D 6D-rot, or 13D 9D-rot)
        cam2ncam: 4x4 transformation matrix from camera to normalized camera space
        intrinsics: 3x3 camera intrinsics matrix
        original_image_pil: PIL Image of the original input image
        image_size: Optional render resolution (defaults to original image size)
        skip_glb_rotation: Whether to skip the 90-degree X-axis rotation applied to GLB files
            - False (default): Apply 90° rotation (use for GLB exports from postprocessing_utils.to_glb)
            - True: Skip rotation (use for meshes from marching cubes or direct mesh extraction)
        pose_representation: Optional pose format override ('quaternion_translation_scale', '6d_translation_scale', '9d_translation_scale')
            If None, auto-detects from tensor shape
    
    Returns:
        overlay_image: PIL Image with mesh composited on original image
        mesh_only_image: PIL Image with only the rendered mesh
    
    Note:
        GLB files exported via postprocessing_utils.to_glb() need a 90° X rotation to align
        with the camera coordinate system. Meshes extracted directly via marching cubes 
        are already correctly oriented and should skip this rotation.
    """
    # Extract vertices from the mesh
    vertices = np.array(mesh.vertices)
    
    # Auto-detect pose representation from tensor shape if not provided
    if pose_representation is None:
        pose_dim = pred_pose_tensor.shape[0]
        if pose_dim == 8:
            pose_representation = 'quaternion_translation_scale'
        elif pose_dim == 10:
            pose_representation = '6d_translation_scale'
        elif pose_dim == 13:
            pose_representation = '9d_translation_scale'
        else:
            raise ValueError(f"Unexpected pose tensor dimension: {pose_dim}. Expected 8, 10, or 13.")
    
    # Parse pose using universal parser
    parsed_pose = parse_pose_output(pred_pose_tensor, pose_representation)
    r_pred = R.from_matrix(parsed_pose['rotation_matrix'])
    scale = parsed_pose['scale']
    translation = parsed_pose['translation']

     # Apply transformations: scale
    vertices = vertices * scale

    # GLB requires special transformation: first rotate 90 degrees around X, then apply pose
    # Meshes from marching cubes (in run_snapshot of ss) are already oriented correctly, so skip this rotation if requested
    if not skip_glb_rotation:
        r90 = R.from_euler('xyz', [90, 0, 0], degrees=True)
        # Apply transformations: scale -> r90 
        vertices = vertices @ r90.as_matrix().T

    # Apply transformations: rotation -> translation
    vertices = vertices @ r_pred.as_matrix().T + translation
    vertices_ncam = vertices

    # Transform from NCAM back to camera (OpenCV) space
    cam2ncam_inv = np.linalg.inv(cam2ncam)
    vertices_cam_h = np.hstack([vertices_ncam, np.ones((vertices_ncam.shape[0], 1))])
    vertices_cam = (vertices_cam_h @ cam2ncam_inv.T)[:, :3]

    # Convert camera coordinates from OpenCV (x right, y down, z forward)
    # to OpenGL camera coordinates expected by Open3D (x right, y up, z backward)
    cv2gl = coord_transform()[:3, :3]
    vertices_gl = (vertices_cam @ cv2gl.T)

    # Create Open3D mesh from GLB
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices_gl)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
    mesh_o3d.compute_vertex_normals()

    # Get vertex colors from GLB if available
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = np.array(mesh.visual.vertex_colors)[:, :3]  # RGB only
        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0
        vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
    else:
        # Fallback to light gray if no colors
        n_verts = len(vertices_gl)
        vertex_colors = np.tile([0.8, 0.8, 0.8], (n_verts, 1))

    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Determine render size
    orig_w, orig_h = original_image_pil.size
    render_w = orig_w if image_size is None else image_size
    render_h = orig_h if image_size is None else image_size

    # Use offscreen renderer for headless rendering (suppress Filament/EGL logs)
    with suppress_stdout_stderr():
        renderer = o3d.visualization.rendering.OffscreenRenderer(render_w, render_h)

    # Validate mesh before rendering to catch degenerate geometry early
    verts_np = np.asarray(mesh_o3d.vertices)
    if len(verts_np) == 0 or not np.all(np.isfinite(verts_np)):
        raise RuntimeError(
            f"Mesh has invalid vertices (count={len(verts_np)}, "
            f"nan={np.isnan(verts_np).any()}, inf={np.isinf(verts_np).any()})"
        )

    # Setup material - use defaultUnlit with vertex colors
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("mesh", mesh_o3d, mat)

    # Set background to transparent (alpha=0) for proper overlay compositing
    renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])

    # Setup camera intrinsics
    K = np.array(intrinsics, dtype=np.float64).copy()
    if image_size is not None:
        # Scale intrinsics if rendering size differs from original
        sx = render_w / float(orig_w)
        sy = render_h / float(orig_h)
        K[0, 0] *= sx  # fx
        K[1, 1] *= sy  # fy
        K[0, 2] *= sx  # cx
        K[1, 2] *= sy  # cy

    # For Open3D rendering, use vertical FOV approach
    fov_y = 2 * np.arctan(render_h / (2 * K[1, 1])) * 180 / np.pi
    aspect = render_w / render_h

    # Set up camera with FOV
    renderer.scene.camera.set_projection(fov_y, aspect, 0.01, 100.0,
                                         o3d.visualization.rendering.Camera.FovType.Vertical)

    # Camera at origin looking down -Z (OpenGL convention)
    center = np.array([0, 0, -0.5])  # Look at mesh center
    eye = np.array([0, 0, 0])  # Camera at origin
    up = np.array([0, 1, 0])  # Y is up
    renderer.scene.camera.look_at(center, eye, up)

    # Render
    rendered_image = renderer.render_to_image()
    rendered_np = np.asarray(rendered_image).astype(np.float32) / 255.0

    # Handle RGBA or RGB rendering
    if rendered_np.shape[2] == 4:
        mesh_mask = rendered_np[:, :, 3:4]  # Alpha channel as mask
        rendered_rgb = rendered_np[:, :, :3]
        mesh_mask = (mesh_mask > 0.5).astype(np.float32)
    else:
        rendered_rgb = rendered_np
        mesh_mask = (rendered_rgb.sum(axis=2) > 0.1).astype(np.float32)[:, :, None]

    # Match original size and ensure RGB format
    if image_size is None:
        original_match = original_image_pil.convert('RGB')
    else:
        original_match = original_image_pil.resize((render_w, render_h), Image.Resampling.LANCZOS).convert('RGB')
    original_np = np.array(original_match).astype(np.float32) / 255.0

    # Composite: where mesh exists, blend 50% original + 50% mesh
    composited = original_np * (1.0 - 0.5 * mesh_mask) + rendered_rgb * mesh_mask * 0.5
    composited = np.clip(composited * 255, 0, 255).astype(np.uint8)

    # Also prepare pure mesh render
    mesh_only = (rendered_rgb * 255).astype(np.uint8)

    return Image.fromarray(composited), Image.fromarray(mesh_only)


def render_mesh_overlay_pyrender(mesh, pred_pose_tensor, cam2ncam, intrinsics, original_image_pil, image_size=None, skip_glb_rotation=False, pose_representation=None):
    """Render mesh overlay using PyRender (better EGL support for Docker).

    This is an alternative to render_mesh_overlay that uses PyRender instead of Open3D.
    PyRender handles headless EGL rendering more reliably in Docker containers.
    """
    # Extract vertices from the mesh
    vertices = np.array(mesh.vertices).copy()
    faces = np.array(mesh.faces).copy()

    # Auto-detect pose representation from tensor shape if not provided
    if pose_representation is None:
        pose_dim = pred_pose_tensor.shape[0]
        if pose_dim == 8:
            pose_representation = 'quaternion_translation_scale'
        elif pose_dim == 10:
            pose_representation = '6d_translation_scale'
        elif pose_dim == 13:
            pose_representation = '9d_translation_scale'
        else:
            raise ValueError(f"Unexpected pose tensor dimension: {pose_dim}. Expected 8, 10, or 13.")

    # Parse pose using universal parser
    parsed_pose = parse_pose_output(pred_pose_tensor, pose_representation)
    r_pred = R.from_matrix(parsed_pose['rotation_matrix'])
    scale = parsed_pose['scale']
    translation = parsed_pose['translation']

    # Apply transformations: scale
    vertices = vertices * scale

    # GLB requires special transformation: first rotate 90 degrees around X, then apply pose
    if not skip_glb_rotation:
        r90 = R.from_euler('xyz', [90, 0, 0], degrees=True)
        vertices = vertices @ r90.as_matrix().T

    # Apply transformations: rotation -> translation
    vertices = vertices @ r_pred.as_matrix().T + translation

    # Transform from NCAM back to camera (OpenCV) space
    cam2ncam_inv = np.linalg.inv(cam2ncam)
    vertices_h = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    vertices_cam = (vertices_h @ cam2ncam_inv.T)[:, :3]

    # Get vertex colors from mesh if available
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = np.array(mesh.visual.vertex_colors)[:, :3]  # RGB only
        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0
        vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
    else:
        vertex_colors = np.tile([0.8, 0.8, 0.8], (len(vertices_cam), 1))

    # Create trimesh with vertex colors for pyrender
    tm = trimesh.Trimesh(vertices=vertices_cam, faces=faces, vertex_colors=(vertex_colors * 255).astype(np.uint8))

    # Convert to pyrender mesh
    pr_mesh = pyrender.Mesh.from_trimesh(tm)

    # Determine render size
    orig_w, orig_h = original_image_pil.size
    render_w = orig_w if image_size is None else image_size
    render_h = orig_h if image_size is None else image_size

    # Setup camera intrinsics
    K = np.array(intrinsics, dtype=np.float64).copy()
    if image_size is not None:
        sx = render_w / float(orig_w)
        sy = render_h / float(orig_h)
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

    # Create pyrender camera from intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Try IntrinsicsCamera if available, otherwise use PerspectiveCamera
    try:
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
    except AttributeError:
        # Fallback: IntrinsicsCamera not available in this pyrender version
        # Use PerspectiveCamera with approximate yfov
        yfov = 2 * np.arctan(render_h / (2 * fy))
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=render_w/render_h, znear=0.01, zfar=100.0)

    # Create scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pr_mesh)

    # Camera pose: PyRender uses OpenGL convention (y-up, z-backward)
    # OpenCV camera is at origin looking down +Z, we need to flip Y and Z
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    scene.add(camera, pose=camera_pose)

    # Add light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    # Render using offscreen renderer
    renderer = pyrender.OffscreenRenderer(render_w, render_h)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    # Process rendered image
    rendered_np = color.astype(np.float32) / 255.0
    
    # Handle both RGB and RGBA outputs
    if rendered_np.shape[2] == 4:
        mesh_mask = rendered_np[:, :, 3:4]
        rendered_rgb = rendered_np[:, :, :3]
    elif rendered_np.shape[2] == 3:
        # No alpha channel - create mask from depth
        rendered_rgb = rendered_np
        mesh_mask = (depth > 0).astype(np.float32)[:, :, np.newaxis]
    else:
        raise ValueError(f"Unexpected color channel count: {rendered_np.shape[2]}")
    
    mesh_mask = (mesh_mask > 0.1).astype(np.float32)

    # Match original size
    if image_size is None:
        original_match = original_image_pil.convert('RGB')
    else:
        original_match = original_image_pil.resize((render_w, render_h), Image.Resampling.LANCZOS).convert('RGB')
    original_np = np.array(original_match).astype(np.float32) / 255.0

    # Composite
    composited = original_np * (1.0 - 0.5 * mesh_mask) + rendered_rgb * mesh_mask * 0.5
    composited = np.clip(composited * 255, 0, 255).astype(np.uint8)
    mesh_only = (rendered_rgb * 255).astype(np.uint8)

    return Image.fromarray(composited), Image.fromarray(mesh_only)


def render_gt_mesh_overlay_pyrender(gt_mesh_path, cam2world, intrinsics, original_image_pil, color=(0, 200, 0)):
    """Render GT mesh overlay on image using PyRender.

    Args:
        gt_mesh_path: Path to GT mesh file (.ply, .obj, etc.)
        cam2world: 4x4 camera-to-world transform (from poses/ files)
        intrinsics: 3x3 camera intrinsics matrix
        original_image_pil: Original image as PIL Image
        color: RGB tuple for GT mesh color (default: green)

    Returns:
        overlay_img: PIL Image with GT mesh composited on original
    """
    gt_mesh = trimesh.load(gt_mesh_path, force='mesh')
    world2cam = np.linalg.inv(cam2world)
    gt_mesh.apply_transform(world2cam)

    vertices_cam = np.array(gt_mesh.vertices)
    faces = np.array(gt_mesh.faces)

    # Color the mesh with the specified color
    vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(vertices_cam), 1))
    tm = trimesh.Trimesh(vertices=vertices_cam, faces=faces, vertex_colors=vertex_colors)
    pr_mesh = pyrender.Mesh.from_trimesh(tm)

    orig_w, orig_h = original_image_pil.size
    K = np.array(intrinsics, dtype=np.float64).copy()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    try:
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
    except AttributeError:
        yfov = 2 * np.arctan(orig_h / (2 * fy))
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=orig_w/orig_h, znear=0.01, zfar=100.0)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pr_mesh)

    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(orig_w, orig_h)
    color_arr, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    rendered_np = color_arr.astype(np.float32) / 255.0
    if rendered_np.shape[2] == 4:
        mesh_mask = rendered_np[:, :, 3:4]
        rendered_rgb = rendered_np[:, :, :3]
    else:
        rendered_rgb = rendered_np
        mesh_mask = (depth > 0).astype(np.float32)[:, :, np.newaxis]
    mesh_mask = (mesh_mask > 0.1).astype(np.float32)

    original_np = np.array(original_image_pil.convert('RGB')).astype(np.float32) / 255.0
    composited = original_np * (1.0 - 0.5 * mesh_mask) + rendered_rgb * mesh_mask * 0.5
    composited = np.clip(composited * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(composited)


def render_mesh_overlay_cv2(mesh, pred_pose_tensor, cam2ncam, intrinsics, original_image_pil, image_size=None, skip_glb_rotation=False, pose_representation=None):
    """Render mesh overlay using OpenCV rasterization - no EGL/GPU context needed.

    Drop-in replacement for render_mesh_overlay. Uses painter's algorithm with
    cv2.fillPoly. Safe for multi-process distributed environments (no Open3D/EGL).
    """
    import cv2

    vertices = np.array(mesh.vertices).copy()
    faces = np.array(mesh.faces).copy()

    # Auto-detect pose representation
    if pose_representation is None:
        pose_dim = pred_pose_tensor.shape[0]
        if pose_dim == 8:
            pose_representation = 'quaternion_translation_scale'
        elif pose_dim == 10:
            pose_representation = '6d_translation_scale'
        elif pose_dim == 13:
            pose_representation = '9d_translation_scale'
        else:
            raise ValueError(f"Unexpected pose tensor dimension: {pose_dim}. Expected 8, 10, or 13.")

    parsed_pose = parse_pose_output(pred_pose_tensor, pose_representation)
    r_pred = R.from_matrix(parsed_pose['rotation_matrix'])
    scale = parsed_pose['scale']
    translation = parsed_pose['translation']

    vertices = vertices * scale
    if not skip_glb_rotation:
        r90 = R.from_euler('xyz', [90, 0, 0], degrees=True)
        vertices = vertices @ r90.as_matrix().T
    vertices = vertices @ r_pred.as_matrix().T + translation

    # Transform from NCAM to camera (OpenCV) space: z forward, y down, x right
    cam2ncam_inv = np.linalg.inv(cam2ncam)
    vertices_h = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    vertices_cam = (vertices_h @ cam2ncam_inv.T)[:, :3]

    orig_w, orig_h = original_image_pil.size
    render_w = orig_w if image_size is None else image_size
    render_h = orig_h if image_size is None else image_size

    K = np.array(intrinsics, dtype=np.float64).copy()
    if image_size is not None:
        sx = render_w / float(orig_w)
        sy = render_h / float(orig_h)
        K[0, 0] *= sx; K[1, 1] *= sy; K[0, 2] *= sx; K[1, 2] *= sy

    # Project to image plane
    z = vertices_cam[:, 2]
    valid = z > 0.001
    z_safe = np.where(valid, z, 1.0)
    u = K[0, 0] * vertices_cam[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * vertices_cam[:, 1] / z_safe + K[1, 2]
    pixels = np.stack([u, v], axis=1)  # (N, 2) x/y

    # Vertex colors
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        vc = np.array(mesh.visual.vertex_colors)[:, :3]
        vertex_colors = vc.astype(np.uint8) if vc.max() > 1.0 else (vc * 255).astype(np.uint8)
    else:
        vertex_colors = np.full((len(vertices), 3), 200, dtype=np.uint8)

    # Face normals for simple two-sided diffuse shading
    v0, v1, v2 = vertices_cam[faces[:, 0]], vertices_cam[faces[:, 1]], vertices_cam[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(nlen > 0, nlen, 1.0)
    diffuse = np.clip(0.3 + 0.7 * np.abs(normals[:, 2]), 0.0, 1.0)  # light along +z (camera axis)

    face_z = z[faces].mean(axis=1)
    face_colors = vertex_colors[faces].mean(axis=1)  # (F, 3)
    face_valid = valid[faces].all(axis=1)

    # Back-to-front sort (painter's algorithm)
    order = np.argsort(-face_z)

    rendered = np.zeros((render_h, render_w, 3), dtype=np.uint8)
    mask = np.zeros((render_h, render_w), dtype=np.uint8)

    for fi in order:
        if not face_valid[fi]:
            continue
        pts = pixels[faces[fi]].astype(np.int32)
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= render_w:
            continue
        if pts[:, 1].max() < 0 or pts[:, 1].min() >= render_h:
            continue
        color = (face_colors[fi] * diffuse[fi]).astype(np.uint8).tolist()
        pts_cv = pts.reshape((-1, 1, 2))
        cv2.fillPoly(rendered, [pts_cv], color)
        cv2.fillPoly(mask, [pts_cv], 255)

    mesh_mask = (mask > 0).astype(np.float32)[:, :, None]
    rendered_rgb = rendered.astype(np.float32) / 255.0

    if image_size is None:
        original_match = original_image_pil.convert('RGB')
    else:
        original_match = original_image_pil.resize((render_w, render_h), Image.Resampling.LANCZOS).convert('RGB')
    original_np = np.array(original_match).astype(np.float32) / 255.0

    composited = original_np * (1.0 - 0.5 * mesh_mask) + rendered_rgb * mesh_mask * 0.5
    composited = np.clip(composited * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(composited), Image.fromarray(rendered)
