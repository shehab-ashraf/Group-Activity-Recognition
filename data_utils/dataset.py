from torch.utils.data import Dataset
import cv2
import torch
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Tuple, Union
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
        transform  = None,
        labels: Dict[str, int] = None,
        split: List[int] = [],
        mode: str = "image_level", 
        only_middle_frame: bool = False,
        crop: bool = False,
        seq: bool = False
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
        """
        self.videos_path = Path(videos_path)
        self.transform = transform
        self.labels = labels or {}
        self.mode = mode
        self.only_middle_frame = only_middle_frame
        self.crop = crop
        self.seq = seq
    

        with open(annot_path, "rb") as f:
            video_annotations = pickle.load(f)

        self.data = self._prepare_dataset(video_annotations, split)

    def _prepare_dataset(
        self, video_annotations: Dict, split: List[int]
    ) -> List[Dict[str, Union[str, int, List]]]:
        
        dataset = []
        for video in split:
            video_data = video_annotations.get(str(video), {})
            for clip_id, clip_metadata in video_data.items():
                frame_boxes = clip_metadata.get("frame_boxes_dct", {})
                category = clip_metadata.get("category", "")
                clip_sequence = []

                for frame_id, boxes in frame_boxes.items():
                    frame_path = self.videos_path / str(video) / clip_id / f"{frame_id}.jpg"
                    if not frame_path.exists(): continue
                    if self.only_middle_frame and str(frame_id) != str(clip_id): continue
                    
                    if self.mode == "player_level":
                        for bbox in boxes:
                            x1, y1, x2, y2 = bbox.box
                            dataset.append({
                                "frame_path": str(frame_path),
                                "category": bbox.category,
                                "x1": x1, "x2": x2, "y1": y1, "y2": y2
                            })
                    
                    elif self.mode == "image_level":
                        if self.seq:
                            clip_sequence.append(str(frame_path))
                        else :
                            bbox_list = [(bbox.box) for bbox in boxes]
                            dataset.append({
                                "frame_path": str(frame_path),
                                "category": category,
                                "bboxes": bbox_list
                            })
                if self.mode == 'image_level' and self.seq and clip_sequence:
                    dataset.append({"sequence": clip_sequence, "category": category})       
        return dataset

    def __len__(self) -> int:
        return len(self.data)
    

    def _load_image(self, path: str) -> torch.Tensor:
        image = cv2.imread(path)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:

        sample = self.data[idx]

        if self.mode == "player_level":
            image = cv2.imread(sample["frame_path"])
            x1, y1, x2, y2 = sample["x1"], sample["y1"], sample["x2"], sample["y2"]
            cropped_image = image[y1:y2, x1:x2]
            if self.transform:
                cropped_image = self.transform(image=cropped_image)["image"]
            return cropped_image, self.labels.get(sample["category"], -1)

        elif self.mode == "image_level":
            if self.seq:
                sequence = [self._load_image(frame) for frame in sample["sequence"]]
                return torch.stack(sequence), self.labels.get(sample["category"], -1)
            elif self.crop:
                image = cv2.imread(sample["frame_path"])
                cropped_images = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in sample["bboxes"]]
                if self.transform:
                    cropped_images = [self.transform(image=cropped_image)["image"] for cropped_image in cropped_images]
                cropped_images += [torch.zeros(3, 224, 224)] * (12 - len(cropped_images))
                return torch.stack(cropped_images), self.labels.get(sample["category"], -1)
            else :
                return self._load_image(sample["frame_path"]), self.labels.get(sample["category"], -1)
