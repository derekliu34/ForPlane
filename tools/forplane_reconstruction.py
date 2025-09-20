# a sp version for forplane recon.
import torch
import numpy as np
import matplotlib.pyplot as plt
# import mcubes
# import trimesh
import os
import configargparse
import open3d as o3d
import cv2
import glob
import imageio


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

'''
Setup
'''

# set render params for DaVinci endoscopic
hwf = [500, 640, 569.46820041]

# set cuda
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
PointCloud reconstruction
'''

###################################################################################################
# Usage Example
###################################################################################################

# python endo_pc_reconstruction.py --config_file configs/example.txt --n_frames 120

###################################################################################################
class PointCloudSequenceVisualizer:
    def __init__(self, pcd_list, stall_count=0, save_dir='render_results', no_autoplay=False, no_loop=False, rec_video_fps=30, cam_move='none'):
        self.pcd_list = pcd_list

        self.stall_count = stall_count
        self.stall_index = 0

        self.playing = not no_autoplay
        self.loop = not no_loop

        self.cam_movement = cam_move
        self.cam_move_params = {
            'swing': {'move_dir': 1, 'move_accum': 0}
        }

        self.recording = False
        self.rec_video_fps = rec_video_fps
        self.rec_buffer = []

        self.frame_idx = 0

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        # 1920 * 1080
        self.vis.create_window(width=1920, height=1536)

        self.save_dir = save_dir

        # initialize point clouds
        self.geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(self.pcd_list[self.frame_idx].points).copy()))
        self.geometry.colors = o3d.utility.Vector3dVector(np.asarray(self.pcd_list[self.frame_idx].colors).copy())
        self.vis.add_geometry(self.geometry)

        # modify point clouds for visualization
        self.vis.get_render_option().point_size = 5

        # Create a light
        # light = o3d.visualization.rendering.PointLight()

        # # Set light properties (like position)
        # light.position = [0, 10, 10]  # The x, y, z position in the 3D space
        # light.color = [1, 1, 1]  # The RGB color of the light
        # light.intensity = 1000  # The intensity/brightness of the light

        # Add light to the scene
        # self.get_render_option().point_lights.append(light)

        key_to_callback = {}
        key_to_callback[ord("A")] = lambda _: self._prev_frame()
        key_to_callback[ord("D")] = lambda _: self._next_frame()
        key_to_callback[ord("P")] = lambda _: self._pause_loop()
        key_to_callback[ord("R")] = lambda _: self._reset_cam_pose()
        key_to_callback[ord("O")] = lambda _: print('Current frame idx:', self.frame_idx)
        key_to_callback[ord("C")] = lambda _: self._save_cam_pose()
        key_to_callback[ord("L")] = lambda _: self._load_cam_pose()
        key_to_callback[ord("S")] = lambda _: self._capture_screenshot()
        key_to_callback[ord("V")] = lambda _: self._video_record()
        for k in key_to_callback:
            self.vis.register_key_callback(k, key_to_callback[k])

        print('####### Manual #######')
        print('A: previous frame')
        print('D: next frame')
        print('P: pause')
        print('R: reset camera pose')
        print('O: output frame index')
        print('C: save camera pose')
        print('L: load camera view')
        print('S: save screenshot')
        print('V: turn on/off screen recording')
        print('######################')

        self.vis.register_animation_callback(lambda _: self._loop_update_cb())

    def _loop_update_cb(self):
        if self.stall_index < self.stall_count:
            self.stall_index += 1
            return False
        else:
            self.stall_index = 0

        self.geometry.points = self.pcd_list[self.frame_idx].points
        self.geometry.colors = self.pcd_list[self.frame_idx].colors

        if self.playing:
            if self.recording:
                frame = self.vis.capture_screen_float_buffer(do_render=True)
                self.rec_buffer.append(frame)

            # self.frame_idx += 1

            # if self.frame_idx >= len(self.pcd_list):
            #     if not self.loop:
            #         self.playing = False
            #         self.frame_idx = len(self.pcd_list) - 1
            #     else:
            #         self.frame_idx = 0
            self._next_frame()
            self._update_camera_movement()

        return True

    def _update_camera_movement(self):
        ctr = self.vis.get_view_control()
        if self.cam_movement == 'swing':
            ctr.rotate(4.0 * self.cam_move_params['swing']['move_dir'], 0.0)
            self.cam_move_params['swing']['move_accum'] = self.cam_move_params['swing']['move_accum'] + self.cam_move_params['swing']['move_dir']

            if abs(self.cam_move_params['swing']['move_accum']) >= 30:
                self.cam_move_params['swing']['move_dir'] = -1 * self.cam_move_params['swing']['move_dir']
                self.cam_move_params['swing']['move_accum'] = 0


    def _reset_cam_pose(self):
        # ctr = self.vis.get_view_control()
        # init_param = ctr.convert_to_pinhole_camera_parameters()
        # init_param.extrinsic = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
        # ctr.convert_from_pinhole_camera_parameters(init_param)

        param = o3d.io.read_pinhole_camera_parameters('cam.json')
        # intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic
        # ctr = self.vis.get_view_control()
        init_param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        init_param.extrinsic = extrinsic
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(init_param)

        self.stall_index = self.stall_count

        return True

    def _video_record(self):
        self.recording = not self.recording

        if not self.recording and len(self.rec_buffer) > 0:
            frames = [np.array(f) for f in self.rec_buffer]
            frames = np.stack(frames, 0)
            self.rec_buffer = []

            imageio.mimwrite(os.path.join(self.save_dir, 'rec_video.mp4'), to8b(frames), fps=self.rec_video_fps, quality=8)
        
        if not self.recording:
            print('Recording stopped. Video saved.')
        else:
            print('Start recording...')

        return True

    def _save_cam_pose(self):
        ctr = self.vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()

        filename = input('Camera pose save path: ')
        o3d.io.write_pinhole_camera_parameters(filename, param)

        return True
    
    def _load_cam_pose(self):
        filename = input('Camera pose load path: ')
        param = o3d.io.read_pinhole_camera_parameters(filename)
        # intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic

        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        init_param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(init_param)

        return True
    
    def _capture_screenshot(self):
        filename = os.path.join(self.save_dir, 'frame_%s.png' % self.frame_idx)
        self.vis.capture_screen_image(filename, do_render=True)

        return True

    def _pause_loop(self):
        self.playing = not self.playing

        if not self.loop and self.frame_idx == len(self.pcd_list) - 1:
            self.playing = False
            self.frame_idx = 0
    
    def _next_frame(self):
        self.frame_idx += 1
        # we save some frames to disk for compare
        if self.frame_idx >= len(self.pcd_list):
            self.frame_idx = 0

        filename = os.path.join(self.save_dir, 'frame_%06d.png' % self.frame_idx)
        self.vis.capture_screen_image(filename, do_render=True)
        print("image saved in %s" % filename)

        self.stall_index = self.stall_count
    
    def _prev_frame(self):
        self.frame_idx -= 1

        if self.frame_idx < 0:
            self.frame_idx = len(self.pcd_list) - 1
        
        self.stall_index = self.stall_count
    
    def run(self):
        self._reset_cam_pose()

        self.vis.run()
        self.vis.destroy_window()


def recon_pcds_from_rgb_d(rgb_np, depth_np, verbose=False, vis_rgbd=False, depth_filter=None, crop_left_size=0):
    if crop_left_size > 0:
        rgb_np = rgb_np[:, crop_left_size:, :]
        depth_np = depth_np[:, crop_left_size:]

    if depth_filter is not None:
        depth_np = cv2.bilateralFilter(depth_np, depth_filter[0], depth_filter[1], depth_filter[2])

        for i in range(150):
            depth_np[:,i] *= (350+i)/500
        depth_np[:,:100] = cv2.GaussianBlur(depth_np[:,:100], (25, 25), 30)
        depth_np[:,:110] = cv2.bilateralFilter(depth_np[:,:110], 9, 75, 75)
        depth_np[:,:120] = cv2.GaussianBlur(depth_np[:,:120], (25, 25), 30)
        depth_np[:,:125] = cv2.bilateralFilter(depth_np[:,:125], 9, 75, 75)
    if verbose:
        print('min depth:', depth_np.min(), 'max depth:', depth_np.max())

    rgb_im = o3d.geometry.Image(rgb_np.astype(np.uint8))
    depth_np = np.ascontiguousarray(depth_np)
    depth_im = o3d.geometry.Image(depth_np)

    depth_np = np.asarray(depth_im)
    plt.imshow(depth_np)
    plt.axis("off")
    plt.show()

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, 
                convert_rgb_to_intensity=False)

    if vis_rgbd:
        plt.subplot(1, 2, 1)
        plt.title('RGB image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(rgbd_image.depth)
        plt.colorbar()
        plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(hwf[1],hwf[0], hwf[2], hwf[2], hwf[1] / 2., hwf[0] / 2.)
        # width, height, fx, fy, cx, cy
    )

    return pcd


if __name__ == '__main__':
    cfg_parser = configargparse.ArgumentParser()
    cfg_parser.add_argument("--input_path", help="input dir path", 
                            default="Forplane_compare_files//Forplane//endonerf_forplane_32k//endonerf_iter32k_gtdp//cutt//estm//rgb")
                            # default="Forplane_compare_files\Forplane\hamlyn_forplane_32k\hamlyn_32k_gt_depth\hamlyn1_32k\estm//rgb")
                            # default="Forplane_compare_files\MForplane\hamlyn_32k_mono_depth\hamlyn1_32k\estm//rgb")
    cfg_parser.add_argument("--no_pc_saved", action='store_true',
                        help='donot save reconstructed point clouds?')
    cfg_parser.add_argument('--out_postfix', type=str, default='',
                        help='the postfix append to the output directory name')
    cfg_parser.add_argument("--vis_rgbd", action='store_true', 
                        help='visualize RGBD output from NeRF?')
    # cfg_parser.add_argument("--start_t", type=float, default=0.0,
    #                     help='time of start frame')
    # cfg_parser.add_argument("--end_t", type=float, default=1.0,
    #                     help='time of end frame')
    cfg_parser.add_argument("--n_frames", type=int, default=1,
                        help='num of frames')
    cfg_parser.add_argument("--depth_smoother", action='store_true',
                        help='apply bilateral filtering on depth maps?')
    cfg_parser.add_argument("--depth_smoother_d", type=int, default=32,
                        help='diameter of bilateral filter for depth maps')
    cfg_parser.add_argument("--depth_smoother_sv", type=float, default=64,
                        help='The greater the value, the depth farther to each other will start to get mixed')
    cfg_parser.add_argument("--depth_smoother_sr", type=float, default=32,
                        help='The greater its value, the more further pixels will mix together')
    cfg_parser.add_argument("--crop_left_size", type=int, default=75,
                        help='the size of pixels to crop')
    # rendering
    cfg_parser.add_argument("--render_all", action='store_true', help='render all using open3d rendering')
    cfg_parser.add_argument('--vis_stall', type=int, default=2,
                        help='control visualization speed (bigger => slower)')
    cfg_parser.add_argument('--data_format', type=str, default='n', choices=['n', 's'])
    cfg_parser.add_argument('--save_dir', type=str, default='./',
                        help='directory for saving screenshots')
    cfg_parser.add_argument("--no_loop", action='store_true', 
                        help='loop playing?')
    cfg_parser.add_argument("--no_autoplay", action='store_true', 
                        help='auto playing?')
    cfg_parser.add_argument('--rec_video_fps', type=int, default=30,
                        help='FPS of video recording')
    cfg_parser.add_argument('--cam_move', type=str, default='none',
                        help='Movement of cameras: none / swing')
    cfg = cfg_parser.parse_args()
    
    # nerf_parser = config_parser()
    # nerf_args = nerf_parser.parse_args(f'--config {cfg.config_file}')

    # if cfg.reload_ckpt:
    #     setattr(nerf_args, 'ft_path', os.path.join(nerf_args.basedir, nerf_args.expname, cfg.reload_ckpt))
    if not os.path.exists(cfg.save_dir):
        os.mkdir(cfg.save_dir)
    img_files = glob.glob(cfg.input_path + "//*.png")

    depth_path = os.path.dirname(cfg.input_path) + "//depth"

    depth_files = glob.glob(depth_path + "//*.png")
    assert len(img_files) == len(depth_files)
    # note that the order is wrong, just sort the files
    img_files = sorted(img_files)
    depth_files = sorted(depth_files)

    # simple the acctual visualization, we use the actual depth_files
    depth_max = 89
    rgbs = []
    depths = []
    for i, j in enumerate(img_files):
        rgbs.append(cv2.cvtColor(cv2.imread(j), cv2.COLOR_BGR2RGB))
        # depths.append(np.load(j)/depth_max)
        depths.append(cv2.imread(depth_files[i], 0).astype(np.float32)/ 255.)
    
    # sp for some black background,
    # np_depth = np.array(depths)
    # median_dp = np.median(np_depth)
    # print(median_dp)
    # for k in depths:
    #     k[k>0.58] = 0.58

    # output directory
    # if not cfg.no_pc_saved:
    #     out_dir = os.path.join(nerf_args.basedir, nerf_args.expname, f"reconstructed_pcds_{epoch}" + (f"_{cfg.out_postfix}" if cfg.out_postfix else ""))
    #     if not os.path.exists(out_dir):
    #         os.makedirs(out_dir)
    #     cfg_parser.write_config_file(cfg, [os.path.join(out_dir, 'args.txt')])

    # build depth filter
    if cfg.depth_smoother:
        depth_smoother = (cfg.depth_smoother_d, cfg.depth_smoother_sv, cfg.depth_smoother_sr)
    else:
        depth_smoother = None

    # reconstruct pointclouds
    print('Reconstructing point clouds...')


    pcds = []
    out_dir = os.path.join((os.path.dirname(cfg.input_path)), "pcds")
    if os.path.exists(out_dir) and len(os.listdir(out_dir))>0:
        pcd_fns = [os.path.join(cfg.pc_dir, fn) for fn in sorted(os.listdir(out_dir)) if fn.endswith('.ply')]
        print('Loading point clouds...')
        for fn in pcd_fns:
            pcd = o3d.io.read_point_cloud(fn)
            pcds.append(pcd)

        print('Total:', len(pcds), 'point clouds loaded.')
    else:
        for rgb, depth in zip(rgbs, depths):
            pcd = recon_pcds_from_rgb_d(rgb, depth, verbose=True, depth_filter=depth_smoother, vis_rgbd=False, crop_left_size=cfg.crop_left_size)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pcds.append(pcd)
        # break
        # save pcds
        

    if cfg.render_all:
        pc_vis = PointCloudSequenceVisualizer(pcds, cfg.vis_stall, cfg.save_dir, cfg.no_autoplay, cfg.no_loop, cfg.rec_video_fps, cfg.cam_move)
        pc_vis.run()



    if not cfg.no_pc_saved:
        print('Saving point clouds...')
        out_dir = os.path.join(os.path.dirname(os.path.dirname(cfg.input_path)), "pcds")
        os.makedirs(out_dir, exist_ok=True)
        for i, pcd in enumerate(pcds):
        
            # Flip it, otherwise the pointcloud will be upside down
            # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            fn = os.path.join(out_dir, f"frame_{i:06d}_pc.ply")
            o3d.io.write_point_cloud(fn, pcd)
        
        print('Point clouds saved to', out_dir)
        

    