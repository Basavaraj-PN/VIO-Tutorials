import numpy as np
import struct
from open3d import *


def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(np_pcd)
    return pcd



def visualize_point_cloud(pcd):
    visualization.draw_geometries([pcd])


t = convert_kitti_bin_to_pcd("/home/omnipotent/Downloads/data_odometry_velodyne/dataset/sequences/00/velodyne/000000.bin")
# visualize_point_cloud(t)

# Save the point cloud as a PCD file
pcd_filename = "output.pcd"
io.write_point_cloud(pcd_filename, t)