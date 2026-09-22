import torch
import imageio.v3 as iio

url = '/home/cl-ment-prigent/Benchmark-Genesis/reconstruction/human_videos/grasping_cup/take_006/take006_grasping_cup_rgb.mp4'
frames = iio.imread(url, plugin="FFMPEG")

device = 'cuda'
grid_size = 10
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

# Initialisation du traitement online
cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)

# Traitement par fenêtres glissantes
for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
    pred_tracks, pred_visibility = cotracker(
        video_chunk=video[:, ind : ind + cotracker.step * 2]
    )

from cotracker.utils.visualizer import Visualizer
vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)