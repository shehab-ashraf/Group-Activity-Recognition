from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils.dataset import Group_Activity_Recognition_Dataset
from typing import Dict, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_dataloader(
    videos_path: str,
    annot_path: str,
    labels: Optional[Dict[str, int]] = None,
    train_split: List[int] = [],
    valid_split: List[int] = [],
    mode: str = "image_level",
    only_middle_frame: bool = False,
    crop: bool = False,
    seq: bool = False,
    batch_size: int = None,
    num_workers: int = 0,
):
    """
    Creates a DataLoader for the Group Activity Recognition Dataset.
    
    Args:
        videos_path (str): Path to the video frames.
        annot_path (str): Path to the annotation pickle file.
        labels (dict, optional): Category-to-label mapping.
        split (list, optional): List of video IDs to include.
        mode (str): Processing mode ('image_level' or 'player_level').
        only_middle_frame (bool): Whether to use only the middle frame.
        crop (bool): Whether to crop frames based on bounding boxes.
        seq (bool): Whether to return a sequence of frames.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker threads for data loading.
    Returns:
        DataLoader: A PyTorch DataLoader For The Train And Talid.
    """
    if train_transforms is None:
        train_transforms = A.Compose([
            A.Resize(224, 224),

            A.RandomBrightnessContrast(p=0.9),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.ColorJitter(brightness=0.2),
                A.GaussNoise()
            ], p=0.2),
            
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ], p=0.2),
            
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            ToTensorV2()
        ])
    if valid_transforms is None:
        valid_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    train_data = Group_Activity_Recognition_Dataset(
        videos_path=videos_path,
        annot_path=annot_path,
        labels=labels,
        split=train_split,
        mode=mode,
        only_middle_frame=only_middle_frame,
        crop=crop,
        seq=seq,
        transform=train_transforms
    )
    validation_data = Group_Activity_Recognition_Dataset(
        videos_path=videos_path,
        annot_path=annot_path,
        labels=labels,
        split=valid_split,
        mode=mode,
        only_middle_frame=only_middle_frame,
        crop=crop,
        seq=seq,
        transform=valid_transforms  
    )
    if len(train_split) == 0:
        train_loader = None
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    if len(valid_split) == 0:
        val_loader = None   
    else:
        val_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    class DataLoaders():
        def __init__(self, train_dataloader, test_dataloader):
            self.train = train_dataloader
            self.valid = test_dataloader
    dls = DataLoaders(train_loader, val_loader)
    return dls