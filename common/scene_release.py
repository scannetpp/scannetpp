from pathlib import Path

class ScannetppScene_Release:
    def __init__(self, scene_id, data_root=None):
        self._scene_id = scene_id
        self.data_root = self.path_or_none(data_root)

    @staticmethod
    def path_or_none(path):
        if path is not None:
            path = Path(path)
        return path

    @property
    def scene_id(self):
        return self._scene_id

    # dir containing all data for this scene
    @property
    def scene_root_dir(self):
        return self.data_root / self._scene_id

    ##### scan assets #####
    @property
    def scans_dir(self):
        '''
        dir containing all scan-related data
        '''
        return self.data_root / self._scene_id / 'scans'
    
    @property
    def pc_dir(self):
        '''
        dir containing 1mm point cloud
        '''
        return self.scans_dir 

    @property
    def scan_pc_path(self):
        '''
        path to point cloud
        '''
        return self.pc_dir / 'pc_aligned.ply'
    
    @property
    def scan_pc_mask_path(self):
        '''
        path to the point cloud mask
        '''
        return self.pc_dir / 'pc_aligned_mask.txt'
    
    @property
    def scan_transformed_poses_path(self):
        '''
        path containing all scanner poses transformed to aligned coordinates
        in a single file
        '''
        return self.pc_dir / 'scanner_poses.json'

    @property
    def mesh_dir(self):
        '''
        dir containing all the meshes and related data
        put meshes in the same dir as 1mm PCs
        '''
        return self.scans_dir 
    
    @property
    def scan_mesh_path(self):
        '''
        path to the mesh
        '''
        return self.mesh_dir / 'mesh_aligned_0.05.ply'
    
    @property
    def scan_mesh_mask_path(self):
        '''
        path to the mesh mask
        '''
        return self.mesh_dir / 'mesh_aligned_0.05_mask.txt'
    
    @property
    def scan_mesh_segs_path(self):
        return self.mesh_dir / f'segments.json'
    
    @property
    def scan_anno_json_path(self):
        return self.mesh_dir / f'segments_anno.json'
    
    @property
    def scan_sem_mesh_path(self):
        return self.mesh_dir / f'mesh_aligned_0.05_semantic.ply'
    
    ######## DSLR ########
    @property
    def dslr_dir(self):
        return self.data_root / self._scene_id / 'dslr'
    
    @property
    def dslr_resized_dir(self):
        return self.dslr_dir / 'resized_images'

    @property
    def dslr_resized_mask_dir(self):
        return self.dslr_dir / 'resized_anon_masks'

    @property
    def dslr_original_dir(self):
        return self.dslr_dir / 'original_images'
    
    @property
    def dslr_original_mask_dir(self):
        return self.dslr_dir / 'original_anon_masks'
    
    @property
    def dslr_resized_undistorted_dir(self):
        return self.dslr_dir / 'resized_undistorted_images'
    
    @property
    def dslr_resized_undistorted_mask_dir(self):
        return self.dslr_dir / 'resized_undistorted_masks'
    
    @property
    def dslr_colmap_dir(self):
        return self.dslr_dir / 'colmap'
    
    @property
    def dslr_nerfstudio_transform_path(self):
        return self.dslr_dir / 'nerfstudio' / 'transforms.json'
    
    @property
    def dslr_nerfstudio_transform_undistorted_path(self):
        return self.dslr_dir / 'nerfstudio' / 'transforms_undistorted.json'
    
    @property
    def dslr_train_test_lists_path(self):
        return self.dslr_dir / 'train_test_lists.json'
    
    ##### iphone #####
    @property
    def iphone_data_dir(self):
        return self.data_root / self._scene_id / 'iphone'
    
    @property
    def iphone_video_path(self):
        return self.iphone_data_dir / 'rgb.mkv'
    
    @property
    def iphone_rgb_dir(self):
        return self.iphone_data_dir / 'rgb'

    @property
    def iphone_video_mask_path(self):
        return self.iphone_data_dir / 'rgb_mask.mkv'
    
    @property
    def iphone_video_mask_dir(self):
        return self.iphone_data_dir / 'rgb_masks'
    
    @property
    def iphone_depth_path(self):
        return self.iphone_data_dir / 'depth.bin'
    
    @property
    def iphone_depth_dir(self):
        return self.iphone_data_dir / 'depth'

    @property
    def iphone_pose_intrinsic_imu_path(self):
        return self.iphone_data_dir / 'pose_intrinsic_imu.json'

    @property
    def iphone_colmap_dir(self):
        return self.iphone_data_dir / 'colmap' 
    
    @property
    def iphone_nerfstudio_transform_path(self):
        return self.iphone_data_dir / 'nerfstudio' / 'transforms.json'
    
    @property
    def iphone_exif_path(self):
        return self.iphone_data_dir / 'exif.json'