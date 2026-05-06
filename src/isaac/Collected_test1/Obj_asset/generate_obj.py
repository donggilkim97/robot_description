import os
import random
import csv
import time
import math

import omni.usd
import omni.kit.commands
import omni.timeline
import omni.kit.app
import numpy as np

try:
    from isaacsim.core.utils.stage import add_reference_to_stage
except Exception:
    from omni.isaac.core.utils.stage import add_reference_to_stage

from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema, UsdShade


# -----------------------------
# User settings
# -----------------------------
folder_path = "/home/dk/robot_description/src/isaac/Collected_test1/Obj_asset"
target_root_path = "/World/TargetObject"

# Ground-truth outputs used by one_shot_grasp.py
ground_truth_csv_path = "/home/dk/robot_description/sfm_dataset/ground_truth_objects.csv"
ground_truth_pcd_dir = "/home/dk/robot_description/sfm_dataset/ground_truth_pcd"

spawn_position = Gf.Vec3d(0.5, 0.0, 0.08)
object_mass = 0.10

# Let the object settle under physics before recording GT.
settle_before_gt_save = True
settle_seconds = 1.5

# GT point cloud export settings
max_gt_pcd_points = 50000

# Back to previous style.
# This keeps collider closer to the actual object shape.
collision_approximation = "convexDecomposition"

static_friction = 1.2
dynamic_friction = 1.0
restitution = 0.0

contact_offset = 0.005
rest_offset = 0.0


# -----------------------------
# Helper functions
# -----------------------------
def remove_api_if_exists(prim, api_schema):
    try:
        if prim.HasAPI(api_schema):
            prim.RemoveAPI(api_schema)
    except Exception as e:
        print(f"[WARN] Could not remove API {api_schema} from {prim.GetPath()}: {e}")


def remove_nested_rigid_bodies(root_prim):
    """
    Remove RigidBodyAPI/MassAPI from all children.
    Keep only ONE rigid body on /World/TargetObject.
    """
    for prim in Usd.PrimRange(root_prim):
        if prim.GetPath() == root_prim.GetPath():
            continue

        remove_api_if_exists(prim, UsdPhysics.RigidBodyAPI)
        remove_api_if_exists(prim, UsdPhysics.MassAPI)

        try:
            remove_api_if_exists(prim, PhysxSchema.PhysxRigidBodyAPI)
        except Exception:
            pass


def create_or_get_scope(stage, path):
    prim = stage.GetPrimAtPath(path)

    if not prim.IsValid():
        UsdGeom.Scope.Define(stage, path)

    return stage.GetPrimAtPath(path)


def create_or_get_physics_material(stage):
    create_or_get_scope(stage, "/World/PhysicsMaterials")

    material_path = "/World/PhysicsMaterials/HighFrictionPhysicsMaterial"
    mat_prim = stage.GetPrimAtPath(material_path)

    if not mat_prim.IsValid():
        material = UsdShade.Material.Define(stage, material_path)
    else:
        material = UsdShade.Material(mat_prim)

    mat_prim = material.GetPrim()

    if not mat_prim.HasAPI(UsdPhysics.MaterialAPI):
        physics_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    else:
        physics_mat = UsdPhysics.MaterialAPI(mat_prim)

    physics_mat.CreateStaticFrictionAttr().Set(static_friction)
    physics_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics_mat.CreateRestitutionAttr().Set(restitution)

    return material


def bind_physics_material_only(prim, material):
    """
    Bind only physics material.
    Visual material / texture is preserved.
    """
    try:
        rel = prim.CreateRelationship("material:binding:physics", False)
        rel.ClearTargets(True)
        rel.AddTarget(material.GetPath())
    except Exception as e:
        print(f"[WARN] Could not bind physics material to {prim.GetPath()}: {e}")


def apply_root_rigid_body(root_prim):
    """
    Apply exactly one dynamic rigid body to the root object.
    """
    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    else:
        rb_api = UsdPhysics.RigidBodyAPI(root_prim)

    rb_api.CreateRigidBodyEnabledAttr().Set(True)

    # Make sure object is dynamic, not kinematic.
    try:
        rb_api.CreateKinematicEnabledAttr().Set(False)
    except Exception:
        pass

    if not root_prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    else:
        mass_api = UsdPhysics.MassAPI(root_prim)

    mass_api.CreateMassAttr().Set(object_mass)

    try:
        if not root_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        else:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI(root_prim)

        if hasattr(physx_rb, "CreateEnableCCDAttr"):
            physx_rb.CreateEnableCCDAttr().Set(True)

        if hasattr(physx_rb, "CreateDisableGravityAttr"):
            physx_rb.CreateDisableGravityAttr().Set(False)

    except Exception as e:
        print(f"[WARN] Optional PhysX rigid body settings failed: {e}")


def apply_colliders_to_meshes(root_prim, physics_material):
    """
    Apply collision to mesh descendants only.
    Do NOT apply RigidBodyAPI to mesh descendants.
    """
    mesh_count = 0

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh_count += 1

        # Mesh children should not be separate rigid bodies.
        remove_api_if_exists(prim, UsdPhysics.RigidBodyAPI)
        remove_api_if_exists(prim, UsdPhysics.MassAPI)

        try:
            remove_api_if_exists(prim, PhysxSchema.PhysxRigidBodyAPI)
        except Exception:
            pass

        # Collision API
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        else:
            collision_api = UsdPhysics.CollisionAPI(prim)

        try:
            collision_api.CreateCollisionEnabledAttr().Set(True)
        except Exception:
            pass

        # Mesh collision approximation
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        else:
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)

        mesh_collision_api.CreateApproximationAttr().Set(collision_approximation)

        # PhysX collision tuning
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        else:
            physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)

        physx_collision_api.CreateContactOffsetAttr().Set(contact_offset)
        physx_collision_api.CreateRestOffsetAttr().Set(rest_offset)

        # Physics material only
        bind_physics_material_only(prim, physics_material)

    print(f"[INFO] Applied colliders to {mesh_count} mesh prim(s).")

    if mesh_count == 0:
        print("[WARN] No mesh prims found under /World/TargetObject.")


def clean_camera_nested_rigidbody(stage):
    camera_rigid_paths = [
        "/World/ur5e_custom_setup/camera_link/Realsense/RSD455"
    ]

    for path in camera_rigid_paths:
        prim = stage.GetPrimAtPath(path)

        if prim.IsValid():
            remove_api_if_exists(prim, UsdPhysics.RigidBodyAPI)
            remove_api_if_exists(prim, UsdPhysics.MassAPI)

            try:
                remove_api_if_exists(prim, PhysxSchema.PhysxRigidBodyAPI)
            except Exception:
                pass

            print(f"[INFO] Removed nested rigid body APIs from camera child: {path}")


def settle_simulation(seconds=1.5):
    """
    Step Isaac Sim for a short time so the dynamic object can settle
    before ground truth is recorded.
    """
    try:
        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()

        if not was_playing:
            timeline.play()

        app = omni.kit.app.get_app()

        # Approximate 60 FPS stepping.
        steps = max(1, int(seconds * 60))

        for _ in range(steps):
            app.update()

        if not was_playing:
            timeline.pause()

        print(f"[INFO] Simulation settled for {seconds:.2f} s before GT capture.")

    except Exception as e:
        print(f"[WARN] Simulation settle failed or was skipped: {e}")


def get_world_bbox(stage, prim_path):
    """
    Compute the world-axis-aligned bounding box of the spawned object.
    """
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        print(f"[ERROR] Cannot compute bbox. Invalid prim: {prim_path}")
        return None

    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy
            ],
            True
        )

        bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()

        min_pt = bbox.GetMin()
        max_pt = bbox.GetMax()

        center = Gf.Vec3d(
            (min_pt[0] + max_pt[0]) / 2.0,
            (min_pt[1] + max_pt[1]) / 2.0,
            (min_pt[2] + max_pt[2]) / 2.0
        )

        dims = Gf.Vec3d(
            max_pt[0] - min_pt[0],
            max_pt[1] - min_pt[1],
            max_pt[2] - min_pt[2]
        )

        return min_pt, max_pt, center, dims

    except Exception as e:
        print(f"[ERROR] Failed to compute world bbox for {prim_path}: {e}")
        return None


def save_points_as_ply(points, save_path, color=(0, 255, 0)):
    """
    Save a simple coloured point cloud as ASCII PLY.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        r, g, b = color

        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")


def export_gt_object_pointcloud(stage, root_prim, object_name):
    """
    Export a sampled ground-truth object surface point cloud in world frame.
    This is better than exporting only mesh vertices because it produces
    a denser and more visually meaningful reference cloud.
    """
    if root_prim is None or not root_prim.IsValid():
        print("[WARN] Cannot export GT point cloud: invalid root prim.")
        return ""

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    triangles = []
    triangle_areas = []

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)

        local_points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()

        if local_points is None or face_counts is None or face_indices is None:
            continue

        local_to_world = xform_cache.GetLocalToWorldTransform(prim)

        world_points = []

        for p in local_points:
            wp = local_to_world.Transform(Gf.Vec3d(p[0], p[1], p[2]))
            world_points.append(
                np.array([float(wp[0]), float(wp[1]), float(wp[2])], dtype=np.float64)
            )

        index_offset = 0

        for count in face_counts:
            if count < 3:
                index_offset += count
                continue

            face = face_indices[index_offset:index_offset + count]
            index_offset += count

            # Fan triangulation for polygon faces.
            for i in range(1, count - 1):
                p0 = world_points[face[0]]
                p1 = world_points[face[i]]
                p2 = world_points[face[i + 1]]

                area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))

                if area > 1e-10:
                    triangles.append((p0, p1, p2))
                    triangle_areas.append(area)

    if len(triangles) == 0:
        print("[WARN] No valid triangles found for GT surface point cloud.")
        return ""

    triangle_areas = np.array(triangle_areas, dtype=np.float64)
    area_sum = float(np.sum(triangle_areas))

    if area_sum <= 1e-12:
        print("[WARN] GT mesh surface area is too small.")
        return ""

    probabilities = triangle_areas / area_sum

    sample_count = min(max_gt_pcd_points, 30000)
    chosen_indices = np.random.choice(
        len(triangles),
        size=sample_count,
        replace=True,
        p=probabilities
    )

    points_world = []

    for tri_idx in chosen_indices:
        p0, p1, p2 = triangles[tri_idx]

        r1 = random.random()
        r2 = random.random()

        sqrt_r1 = math.sqrt(r1)

        # Uniform triangle surface sampling.
        a = 1.0 - sqrt_r1
        b = sqrt_r1 * (1.0 - r2)
        c = sqrt_r1 * r2

        p = a * p0 + b * p1 + c * p2
        points_world.append((float(p[0]), float(p[1]), float(p[2])))

    timestamp = int(time.time())
    safe_name = os.path.splitext(object_name)[0].replace(" ", "_")

    save_path = os.path.join(
        ground_truth_pcd_dir,
        f"gt_{timestamp}_{safe_name}.ply"
    )

    save_points_as_ply(
        points=points_world,
        save_path=save_path,
        color=(0, 255, 0)
    )

    print(f"[GT_PCD] Saved sampled GT surface point cloud: {save_path}")
    print(f"[GT_PCD] Number of GT sampled points: {len(points_world)}")

    return save_path


def save_ground_truth(stage, object_name, usd_path, prim_path, spawn_position, yaw_deg, gt_pcd_path=""):
    """
    Save ground-truth object pose and bounding box to CSV.

    one_shot_grasp.py reads the latest row from:
    /home/dk/robot_description/sfm_dataset/ground_truth_objects.csv
    """
    bbox_result = get_world_bbox(stage, prim_path)

    if bbox_result is None:
        print("[ERROR] Ground-truth save failed because bbox could not be computed.")
        return

    min_pt, max_pt, center, dims = bbox_result

    os.makedirs(os.path.dirname(ground_truth_csv_path), exist_ok=True)

    file_exists = os.path.exists(ground_truth_csv_path)

    fieldnames = [
        "timestamp",
        "object_name",
        "usd_path",
        "prim_path",

        "spawn_x",
        "spawn_y",
        "spawn_z",

        "bbox_center_x",
        "bbox_center_y",
        "bbox_center_z",

        "bbox_min_x",
        "bbox_min_y",
        "bbox_min_z",

        "bbox_max_x",
        "bbox_max_y",
        "bbox_max_z",

        "dim_x",
        "dim_y",
        "dim_z",

        "top_z",
        "yaw_deg",
        "yaw_rad",
        "object_mass",
        "gt_pcd_path"
    ]

    # If the old CSV has a different header, back it up and create a new one.
    if file_exists:
        try:
            with open(ground_truth_csv_path, "r") as f:
                existing_header = f.readline().strip().split(",")

            if existing_header != fieldnames:
                backup_path = ground_truth_csv_path + f".backup_{int(time.time())}"
                os.rename(ground_truth_csv_path, backup_path)
                print(f"[GT] Existing GT CSV header changed. Backed up old file to: {backup_path}")
                file_exists = False

        except Exception as e:
            print(f"[WARN] Could not check GT CSV header: {e}")

    with open(ground_truth_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": time.time(),
            "object_name": object_name,
            "usd_path": usd_path,
            "prim_path": prim_path,

            "spawn_x": float(spawn_position[0]),
            "spawn_y": float(spawn_position[1]),
            "spawn_z": float(spawn_position[2]),

            "bbox_center_x": float(center[0]),
            "bbox_center_y": float(center[1]),
            "bbox_center_z": float(center[2]),

            "bbox_min_x": float(min_pt[0]),
            "bbox_min_y": float(min_pt[1]),
            "bbox_min_z": float(min_pt[2]),

            "bbox_max_x": float(max_pt[0]),
            "bbox_max_y": float(max_pt[1]),
            "bbox_max_z": float(max_pt[2]),

            "dim_x": float(dims[0]),
            "dim_y": float(dims[1]),
            "dim_z": float(dims[2]),

            "top_z": float(max_pt[2]),
            "yaw_deg": float(yaw_deg),
            "yaw_rad": float(math.radians(yaw_deg)),
            "object_mass": float(object_mass),
            "gt_pcd_path": gt_pcd_path
        })

    print(f"[GT] Saved ground truth to: {ground_truth_csv_path}")
    print(
        f"[GT] object={object_name}, "
        f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), "
        f"top_z={max_pt[2]:.4f}, "
        f"yaw={yaw_deg:.2f} deg"
    )


def delete_old_target_object(stage):
    if stage.GetPrimAtPath(target_root_path).IsValid():
        omni.kit.commands.execute(
            "DeletePrims",
            paths=[target_root_path]
        )
        print(f"[INFO] Deleted old {target_root_path}")


def spawn_random_object():
    valid_extensions = [".usd", ".usda", ".usdc"]

    object_files = [
        f for f in os.listdir(folder_path)
        if any(f.endswith(ext) for ext in valid_extensions)
    ]

    if not object_files:
        print("Error: No USD files found in the specified folder!")
        return

    stage = omni.usd.get_context().get_stage()

    delete_old_target_object(stage)

    chosen_file = random.choice(object_files)
    usd_path = os.path.join(folder_path, chosen_file)

    add_reference_to_stage(
        usd_path=usd_path,
        prim_path=target_root_path
    )

    root_prim = stage.GetPrimAtPath(target_root_path)

    if not root_prim.IsValid():
        print(f"[ERROR] Failed to spawn object at {target_root_path}")
        return

    # Set transform on root only.
    xformable = UsdGeom.Xformable(root_prim)
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(spawn_position)

    rotate_op = xformable.AddRotateZOp()
    random_yaw = random.uniform(0.0, 360.0)
    rotate_op.Set(random_yaw)

    # Remove nested rigid bodies from referenced asset.
    remove_nested_rigid_bodies(root_prim)

    # Create physics material.
    physics_material = create_or_get_physics_material(stage)

    # Apply one dynamic rigid body to root.
    apply_root_rigid_body(root_prim)

    # Apply object-shape mesh colliders.
    apply_colliders_to_meshes(root_prim, physics_material)

    # Optional cleanup.
    clean_camera_nested_rigidbody(stage)

    # Let object settle before recording GT.
    if settle_before_gt_save:
        settle_simulation(settle_seconds)

    # Re-read root prim after physics stepping.
    root_prim = stage.GetPrimAtPath(target_root_path)

    # Save GT object point cloud for RViz and comparison.
    gt_pcd_path = export_gt_object_pointcloud(
        stage=stage,
        root_prim=root_prim,
        object_name=chosen_file
    )

    # Save ground-truth pose and bounding box for quantitative evaluation.
    save_ground_truth(
        stage=stage,
        object_name=chosen_file,
        usd_path=usd_path,
        prim_path=target_root_path,
        spawn_position=spawn_position,
        yaw_deg=random_yaw,
        gt_pcd_path=gt_pcd_path
    )

    print("--------------------------------------------------")
    print(f"[SUCCESS] Spawned: {chosen_file}")
    print(f"[PATH]    {usd_path}")
    print(f"[POSE]    position={spawn_position}, yaw={random_yaw:.2f} deg")
    print("[PHYSICS] one dynamic root rigid body + mesh child colliders")
    print(f"[COLLIDER] approximation={collision_approximation}")
    print(f"[MASS]     {object_mass} kg")
    print(f"[CONTACT]  contact_offset={contact_offset}, rest_offset={rest_offset}")
    print("[MATERIAL] physics material only; visual texture preserved")
    print(f"[GT_PCD]   {gt_pcd_path}")
    print("--------------------------------------------------")


spawn_random_object()
