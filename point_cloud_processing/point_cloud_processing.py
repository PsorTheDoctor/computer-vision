import open3d as o3d


def voxel_grid(input_pcd):
    voxels = input_pcd.voxel_down_sample(voxel_size=1.0)
    return voxels


def outlier_removal(input_pcd):
    pcd, idx = input_pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.01)
    return pcd  # input_pcd.select_by_index(idx)


def spatial_filter(input_pcd):
    filtered_pcd = input_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-float('inf'), -float('inf'), 260.0),
        max_bound=(float('inf'), float('inf'), 320.0)
    ))
    return filtered_pcd


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud('../data/cloud.pcd')
    o3d.visualization.draw_geometries(
        [pcd], window_name='Point cloud before filtering'
    )
    filtered_pcd = voxel_grid(pcd)
    filtered_pcd = outlier_removal(filtered_pcd)
    filtered_pcd = spatial_filter(filtered_pcd)

    o3d.visualization.draw_geometries(
        [filtered_pcd], window_name='Point cloud after filtering'
    )
