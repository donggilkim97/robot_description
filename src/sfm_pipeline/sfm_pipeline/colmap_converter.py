import json
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def invert_transform(tx, ty, tz, qx, qy, qz, qw):
    r_ros = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    
    R_cv_to_ros = np.array([
        [ 0.0,  0.0,  1.0],
        [-1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0]
    ])
    
    r_cv = np.dot(r_ros, R_cv_to_ros)
    t_vec = np.array([tx, ty, tz])
    
    r_colmap = r_cv.T
    t_colmap = -np.dot(r_colmap, t_vec)
    q_colmap = Rotation.from_matrix(r_colmap).as_quat()
    
    return t_colmap[0], t_colmap[1], t_colmap[2], q_colmap[0], q_colmap[1], q_colmap[2], q_colmap[3]

def main(args=None):
    dataset_dir = "sfm_dataset"
    json_path = os.path.join(dataset_dir, "transforms.json")
    colmap_dir = os.path.join(dataset_dir, "colmap_data")
    img_dir = os.path.join(dataset_dir, "images")
    os.makedirs(colmap_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        poses = json.load(f)
        
    cameras_path = os.path.join(colmap_dir, "cameras.txt")
    images_path = os.path.join(colmap_dir, "images.txt")
    points_path = os.path.join(colmap_dir, "points3D.txt")
    
    sample_img_name = list(poses.keys())[0]
    sample_img = cv2.imread(os.path.join(img_dir, sample_img_name))
    height, width = sample_img.shape[:2]
    
    fx = 1536.0
    fy = 1536.0
    cx = width / 2.0
    original_height = height * (6.0 / 5.0)
    cy = original_height / 2.0
    
    with open(cameras_path, 'w') as f:
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
        
    with open(images_path, 'w') as f:
        image_id = 1
        for img_name, data in sorted(poses.items()):
            tx, ty, tz = data["translation"]
            qx, qy, qz, qw = data["rotation"]
            
            t_inv_x, t_inv_y, t_inv_z, q_inv_x, q_inv_y, q_inv_z, q_inv_w = invert_transform(tx, ty, tz, qx, qy, qz, qw)
            
            f.write(f"{image_id} {q_inv_w} {q_inv_x} {q_inv_y} {q_inv_z} {t_inv_x} {t_inv_y} {t_inv_z} 1 {img_name}\n\n")
            image_id += 1
            
    open(points_path, 'w').close()

if __name__ == '__main__':
    main()