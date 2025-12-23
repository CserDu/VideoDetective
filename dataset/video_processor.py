import random,math
import torch
from PIL import Image
from torch import nn
import decord
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler, UniformClipSampler
from torchvision.transforms.functional import InterpolationMode

video_specaug_params = {
    "mask_rate": 0.0,
}


def crop_boxes(boxes, x_offset, y_offset):
    """
    Peform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to peform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


class VideoProcessor:
    def __init__(
        self,
        sample_per_clip: int = 2,
        clip_duration: int = 1,
    ):
        
        self.frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=sample_per_clip)
        self.clip_sampler = UniformClipSampler(
            clip_duration=clip_duration, backpad_last=True
        )

        self.fps = sample_per_clip

        self.video_transform = transforms.Compose(
            [
                ResizeVideo((224, 224), interpolation_mode="bicubic"),
                NormalizeVideo(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    
    def __call__(self,video_path, shot_duration = None):
        if shot_duration is not None:
            return self.process_with_duration(video_path,shot_duration)
        return self.process_without_duration(video_path)


    def process_with_duration(self,video_path, shot_duration):
        fps = self.fps
        try:
            vr = decord.VideoReader(video_path,width=224,height=224)
            total_frames, video_fps = len(vr), vr.get_avg_fps()
            nframes = int(total_frames / video_fps * fps)
            nframes = max(2, nframes)
            # nframes = min(nframes, 4 * 60 * 2)
            idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
            # print(idx)
            # idx = idx[: 4 * 60 * 2]  # max 4 min.
            video = vr.get_batch(idx).asnumpy() # t,h,w,c
            video = video / 255.
            video = torch.tensor(video,dtype=torch.float32).permute(3, 0, 1, 2)  # C,T,H,W
        except:
            print('===== video process error, video path: ',video_path)
            video = torch.zeros((3,10,224,224),dtype=torch.float32)

        video = self.video_transform(video).transpose(0,1) # t,c,h,w
        t = video.shape[0]
        seg_nums = int(t // (shot_duration * fps))
        if t % (shot_duration * fps) !=0:
            seg_nums += 1
        video_inputs = []
        for i in range(seg_nums):
            st = i * shot_duration * fps
            et = (i+1) * shot_duration * fps
            video_inputs.append(video[st:et])
        return video_inputs
    

    def process_without_duration(self,video_path):
        try:
            vr = decord.VideoReader(video_path,width=224,height=224)
            total_frames, video_fps = len(vr), vr.get_avg_fps()
            duration = total_frames / video_fps
            if duration > 3 * 60 + 30:
                print(f'video_path: {video_path} duration: {duration}')
            fps = self.fps
            nframes = int(total_frames / video_fps * fps)
            nframes = max(2, nframes)
            idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
            video = vr.get_batch(idx).asnumpy() # t,h,w,c
            video = video / 255.
            video = torch.tensor(video,dtype=torch.float32).permute(3, 0, 1, 2)  # C,T,H,W
        except:
            print('===== video process error, video path: ',video_path)
            video = torch.randn((3,20,224,224),dtype=torch.float32)

        video = self.video_transform(video).transpose(0,1) # t,c,h,w
        seg_nums = 1
        video_inputs = []
        for i in range(seg_nums):
            video_inputs.append(video)
        return video_inputs
    

    def get_clip_timepoints(self, clip_sampler, duration):
        # Read out all clips in this video
        all_clips_timepoints = []
        is_last_clip = False
        end = 0.0
        while not is_last_clip:
            start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
            all_clips_timepoints.append((start, end))
        return all_clips_timepoints


class ImageProcessor:
    def __init__(self):
        self.data_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    

    def __call__(self,image_path):
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
        image = self.data_transform(image)
        return image


