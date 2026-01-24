from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
import torch

class MySegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1) Read image
        img = cv2.imread(self.image_paths[idx])            # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # RGB

        # 2) Read mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 3) Apply transforms (optional)
        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img = out["image"]
            mask = out["mask"]
        else:
            # convert to tensor if no transforms
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
            mask = torch.tensor(mask).long()

        return img, mask

if __name__ == "__main__":
    import os
    img_folder = '/Users/akhilgattu/Desktop/VLM_project/data_idrid_multiclass/train/images/'
    mask_folder = '/Users/akhilgattu/Desktop/VLM_project/data_idrid_multiclass/train/masks/'

    img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder)]
    mask_paths = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder)]

    seg_dataset = MySegmentationDataset(
        image_paths=img_paths,
        mask_paths=mask_paths,
        transform=None,
    )


    seg_data_loader = DataLoader(
        dataset=seg_dataset,
        batch_size=5,
        shuffle=False,
        num_workers=5,
    )

    for imgs, masks in seg_data_loader:
        print(f"Image batch size: {imgs.size()}")
        print(f"Mask batch size: {masks.size()}")
