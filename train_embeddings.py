import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from model import (
    SiameseNet, ContrastiveLoss, compose_train_transforms, compose_valid_transforms,
    resize_size, padded_size
)
import numpy as np
import cv2 as cv
import time


def train_epoch(train_loader, nn, loss_fn, optimizer):
    nn.train()
    train_loss = 0
    for batch_idx, (cls, x1, x2) in enumerate(train_loader):
        start = time.time()

        optimizer.zero_grad()
        y1, y2 = nn(x1, x2)

        loss = loss_fn(y1, y2, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(
            f"Train Batch {batch_idx + 1} of {len(train_loader)}, "
            f"loss: {loss}, time: {time.time() - start}"
        )

    train_loss /= len(train_loader)
    print(f"Train Loss: {train_loss}")
    return train_loss


def valid_epoch(val_loader, nn, loss_fn):
    nn.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (cls, x1, x2) in enumerate(val_loader):
            y1, y2 = nn(x1, x2)
            loss = loss_fn(y1, y2, cls)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Valid Loss: {val_loss}")
    return val_loss


def fit(
        train_loader,
        valid_loader,
        nn,
        loss_function,
        optimizer,
        scheduler,
        num_epochs,
        trained_model_path
):
    lowest_val_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1} ...")
        start = time.time()

        _ = train_epoch(train_loader, nn, loss_function, optimizer)
        val_loss = valid_epoch(valid_loader, nn, loss_function)

        scheduler.step()

        if val_loss < lowest_val_loss:
            print(f"Best val loss: {val_loss} ; saving model {trained_model_path}.")
            lowest_val_loss = val_loss
            torch.save(nn.embedding_net().state_dict(), trained_model_path)

        print(f"Done Epoch {epoch + 1} in {(time.time() - start) / 60}")


class AugmentedSiameseDataset(Dataset):
    def __init__(self, dataset, transform_1=None, transform_2=None):
        self.dataset = dataset
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __getitem__(self, index):
        c, x1p, x2p = self.dataset[index]
        x1 = cv.cvtColor(cv.imread(x1p), cv.COLOR_BGR2RGB)
        x2 = cv.cvtColor(cv.imread(x2p), cv.COLOR_BGR2RGB)
        if self.transform_1:
            x1 = self.transform_2(x1)
        if self.transform_2:
            x2 = self.transform_2(x2)
        return c, x1, x2

    def __len__(self):
        return len(self.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--draw", type=bool, default=None)

    parser.set_defaults()
    args = parser.parse_args()

    def dataset_name(name: str):
        return os.path.join(args.root_folder, name + '.pairs.dataset')

    dataset_train = torch.load(dataset_name("train"))
    dataset_valid = torch.load(dataset_name("valid"))

    os.makedirs(os.path.join(args.root_folder, "trained_models"), exist_ok=True)
    embeddings_trained_model_path = os.path.join(
        args.root_folder, "trained_models", "embeddings.pth"
    )

    train_transforms_1 = compose_train_transforms(resize_size, padded_size)
    train_transforms_2 = compose_train_transforms(resize_size, padded_size)

    valid_transforms_1 = compose_valid_transforms(resize_size, padded_size)
    valid_transforms_2 = compose_valid_transforms(resize_size, padded_size)

    dataset_train = AugmentedSiameseDataset(dataset_train, train_transforms_1, train_transforms_2)
    dataset_valid = AugmentedSiameseDataset(dataset_valid, valid_transforms_1, valid_transforms_2)

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True)

    if args.draw:
        go_train = False
        for datapoint in loader_train:
            def draw_image(c, im, imn):
                im = np.transpose(
                    (im.detach().numpy() * 255).astype(np.uint8), (1, 2, 0)
                )[:, :, ::-1]
                if c.numpy().astype(np.uint8) == 0:
                    t = "NEG"
                else:
                    t = "POS"
                im = im.copy()
                cv.putText(im, t, (20, 20),
                           cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1
                )
                cv.imshow(f"datapoint {imn}", im)

            for b in range(args.batch_size):
                c = datapoint[0][b]
                i1 = datapoint[1][b]
                i2 = datapoint[2][b]
                draw_image(c, i1, 1)
                draw_image(c, i2, 2)
                if cv.waitKey() == ord('x'):
                    go_train = True
                    cv.destroyAllWindows()
                    break

            if go_train:
                break

    cuda = torch.cuda.is_available()
    embeddings_trained_model_path_load = embeddings_trained_model_path \
        if os.path.isfile(embeddings_trained_model_path) else None
    nn = SiameseNet(embeddings_size=256,
                    embeddings_weights=embeddings_trained_model_path_load,
                    cuda=cuda)

    margin = 2.0
    loss_function = ContrastiveLoss(margin, cuda)
    lr = 1e-3
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)

    fit(
        loader_train,
        loader_valid,
        nn,
        loss_function,
        optimizer,
        scheduler,
        args.epochs,
        embeddings_trained_model_path
    )


if __name__ == "__main__":
    main()
