import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import optim
from tqdm import tqdm
import torch.nn as nn
from Projekt.unet import UNET
from Projekt.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

LEARNING_RATE = 1e-4
DEVICE = 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 218
IMAGE_WIDTH = 226
PIN_MEMORY = True
LOAD_MODEL = False

basepath = 'Plant_Phenotyping_Dataset'
images = sorted([os.path.join(basepath,f) for f in os.listdir(basepath) if f.endswith('_rgb.png')])
labels = sorted([os.path.join(basepath,f) for f in os.listdir(basepath) if f.endswith('_label.png')])

def train(loader, model, optimizer, loss, scaler):
    #Progress bar
    loop = tqdm(loader)
    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        #forward (float16 training)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss(predictions, targets)

            #backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #update tqdm loop
            loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(inchannels=3, outchannels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        images,
        labels,
        images[:20],
        labels[:20],
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE)


if __name__ == '__main__':
    main()