from torch.utils.data import Dataset
import cv2
import torch
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Tuple, Union

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
        crop: bool = False
    ):
        """
        Args:
            videos_path (str): Path to the folder containing the video frames.
            annot_path (str): Path to the pickle file containing annotations.
            transform (albumentations.Compose, optional): Transformations to apply to the frames.
            labels (dict, optional): A dictionary mapping category names to numeric labels.
            split (list, optional): A list of clip IDs to include in the dataset.
            mode (str): Determines if it's 'image_level' or 'player_level' processing.
            only_middle_frame (bool): If True, only the middle frame of each clip is considered.
            crop (bool): If True, crop the frames based on bounding boxes.
        """
        self.videos_path = Path(videos_path)
        self.transform = transform
        self.labels = labels if labels is not None else {}
        self.mode = mode
        self.only_middle_frame = only_middle_frame
        self.crop = crop

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
                        # Handle image-level (whole frame) logic
                        bbox_list = [(bbox.box) for bbox in boxes]
                        dataset.append({
                            "frame_path": str(frame_path),
                            "category": category,
                            "bboxes": bbox_list
                        })
        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:

        sample = self.data[idx]
        frame_path = sample["frame_path"]
        category = sample["category"]
        image = cv2.imread(frame_path)

        if self.mode == "player_level":
            x1, y1, x2, y2 = sample["x1"], sample["y1"], sample["x2"], sample["y2"]
            cropped_image = image[y1:y2, x1:x2]
            if self.transform:
                cropped_image = self.transform(image=cropped_image)["image"]
            return cropped_image, self.labels[category]
        
        elif self.mode == "image_level":
            if self.crop:
                 cropped_images = []
                 for bbox in sample["bboxes"]:
                     x1, y1, x2, y2 = bbox
                     cropped_image = image[y1:y2, x1:x2]
                     if self.transform:
                        cropped_image = self.transform(image=cropped_image)["image"]
                     cropped_images.append(cropped_image)
                 while len(cropped_images) != 12:
                     cropped_image.append(torch.zeros(3, 224, 224))
                 return torch.stack(cropped_images), self.labels[category]
            else :
                if self.transform:
                    image = self.transform(image=image)["image"]
                return image, self.labels[category]
