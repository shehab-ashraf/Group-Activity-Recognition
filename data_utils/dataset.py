from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union
import pickle
import cv2
import torch
from torch.utils.data import Dataset
import sys

class BoxInfo:
    def __init__(self, line):
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated

sys.modules['boxinfo'] = sys.modules[__name__]

class Group_Activity_Recognition_Dataset(Dataset):
    def __init__(
        self,
        videos_path: str,
        annot_path: str,
        transform=None,
        labels: Dict[str, int] = None,
        split: List[int] = [],
        mode: str = "image_level", 
        only_middle_frame: bool = False,
        crop: bool = False,
        seq: bool = False,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            videos_path (str): Path to the folder containing the video frames.
            annot_path (str): Path to the pickle file containing annotations.
            transform (albumentations.Compose, optional): Transformations to apply to the frames.
            labels (dict, optional): A dictionary mapping category names to numeric labels.
            split (list, optional): A list of videos to include in the dataset.
            mode (str): Determines if it's 'image_level' or 'player_level' processing.
            only_middle_frame (bool): If True, only the middle frame of each clip is considered.
            crop (bool): If True, crop the frames based on bounding boxes.
            seq (bool): If True, return a sequence of frames instead of a single frame.
            target_size (tuple): Target size for output images (height, width).
        """
        self.videos_path = Path(videos_path)
        self.transform = transform
        self.labels = labels or {}
        self.split = split
        self.target_size = target_size
        self.mode = mode
        self.only_middle_frame = only_middle_frame
        self.crop = crop
        self.seq = seq


        with open(annot_path, "rb") as f:
            video_annotations = pickle.load(f)

        self.data = self._prepare_data(video_annotations, split)

    def _prepare_data(self, annotations, split):
        dataset = []
        for video_id in split:
            video_data = annotations.get(str(video_id), {})
            for clip_id, clip_meta in video_data.items():
                frames_data = clip_meta.get("frame_boxes_dct", {})
                clip_category = clip_meta.get("category", "")

                if self.mode == "player_level":
                    dataset.extend(self._process_player_level(video_id, clip_id, frames_data))
                else:
                    dataset.extend(self._process_image_level(video_id, clip_id, frames_data, clip_category))
        return dataset

    def _process_player_level(self, video_id, clip_id, frames_data):
        players_data = defaultdict(list)
        samples = []

        for frame_id, boxes in frames_data.items():
            if not self._should_include_frame(clip_id, frame_id):
                continue

            frame_path = self._get_frame_path(video_id, clip_id, frame_id)
            if not Path(frame_path).exists(): continue
            

            for bbox in boxes:
                if self.seq:
                    players_data[bbox.player_ID].append((frame_path, bbox))
                else:
                    samples.append({
                        'type': 'player',
                        'frame_path': frame_path,
                        'box': bbox.box,
                        'category': bbox.category
                    })

        if self.seq:
            for player_id, sequence in players_data.items():
                samples.append({
                    'type': 'player_sequence',
                    'sequence': sequence,
                    'category': sequence[-1][1].category  # Last frame's label
                })
        return samples

    def _process_image_level(self, video_id, clip_id, frames_data, clip_category):
        samples = []
        sequence_frames = []
        sequence_bboxes = []

        for frame_id, boxes in frames_data.items():
            if not self._should_include_frame(clip_id, frame_id):
                continue

            frame_path = self._get_frame_path(video_id, clip_id, frame_id)
            if not Path(frame_path).exists(): continue
            

            if self.seq:
                sequence_frames.append(frame_path)
                sequence_bboxes.append(boxes)
            else:
                samples.append({
                    'type': 'image',
                    'frame_path': frame_path,
                    'category': clip_category,
                    'bboxes': boxes if self.crop else None
                })

        if self.seq and sequence_frames:
            samples.append({
                'type': 'image_sequence',
                'sequence': sequence_frames,
                'bboxes': sequence_bboxes if self.crop else None,
                'category': clip_category
            })
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if sample['type'] == 'player_sequence' or sample['type'] == 'player':
            return self._load_player_level(sample)
        else:
            return self._load_image_level(sample)
        
    
    def _load_image_level(self, sample):
        if self.seq:
            if self.crop:
                # For sequences with cropping
                sequence = []
                for frame_path, boxes in zip(sample['sequence'], sample['bboxes']):
                    frame = self._load_frame(frame_path, apply_transform=False)
                    crops = (torch.stack([self._crop_frame(frame, bbox.box, apply_transform=True) for bbox in boxes]))
                    if len(crops) < 12:
                        crops += [torch.zeros(3, *self.target_size)] * (12 - len(crops))  # Pad sequence
                    sequence.append(crops)
                return torch.stack(sequence), self.labels.get(sample['category'], -1)
            else:
                # For full-frame sequences
                frames = [self._load_frame(path, apply_transform=True) 
                        for path in sample['sequence']]
                return torch.stack(frames), self.labels.get(sample['category'], -1)
        else:
            if self.crop:
                # Single frame with crops
                frame = self._load_frame(sample['frame_path'], apply_transform=False)
                crops = []
                for bbox in sample['bboxes']:
                    crops.append(self._crop_frame(frame, bbox.box, apply_transform=True))
                if len(crops) < 12:
                    crops += [torch.zeros(3, *self.target_size)] * (12 - len(crops))  # Pad sequence
                return torch.stack(crops), self.labels.get(sample['category'], -1)
            else:
                # Single full frame
                return self._load_frame(sample['frame_path'], apply_transform=True), self.labels.get(sample['category'], -1)

    def _load_player_level(self, sample):
        if self.seq:
            frames = []
            for frame_path, bbox in sample['sequence']:
                frame = self._load_frame(frame_path, apply_transform=False)
                frames.append(self._crop_frame(frame, bbox.box, apply_transform=True))
            # Pad sequence
            frames += [torch.zeros(3, *self.target_size)] * (9 - len(frames))
            return torch.stack(frames[:9]), self.labels.get(sample['category'], -1)
        else:
            frame = self._load_frame(sample['frame_path'], apply_transform=False)
            return self._crop_frame(frame, sample['box'], apply_transform=True), self.labels.get(sample['category'], -1)
    
    def _load_frame(self, path, apply_transform=False):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if apply_transform and self.transform:
            transformed = self.transform(image=img)
            return transformed['image']
        return img
    
    def _crop_frame(self, frame, box, apply_transform=True):
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if apply_transform and self.transform:
            transformed = self.transform(image=crop)
            crop = transformed['image']
            return crop
        return crop
        
    def _should_include_frame(self, clip_id, frame_id):
        return not self.only_middle_frame or str(frame_id) == str(clip_id)

    def _get_frame_path(self, video_id, clip_id, frame_id):
        return self.videos_path / str(video_id) / str(clip_id) / f"{frame_id}.jpg"