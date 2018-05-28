import sys
import time
import datetime

import numpy as np
import scipy.misc
import torch
from torchvision.utils import save_image

from data_classes import *
from conv_autoencoder_dtd_model import DTDAutoEncoder
from cyclical_lr import CyclicLR

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def inference(model, datasource, device, n_subset=None):
    model.eval()
    images = []
    if isinstance(datasource, DataLoader):
        if n_subset is not None:
            subset_count = 0
        for batch_idx, (inputs, scores, _, labels) in enumerate(datasource):
            for i in range(len(inputs)):
                images.append(inputs[i])
                if n_subset is not None:
                    subset_count += 1
                    if subset_count == n_subset:
                        break
            if subset_count == n_subset:
                break
    else:
        if not isinstance(datasource, list):
            datasource = [datasource]
        for d in datasource:
            images.append(TextureDataset.inference_transform()(np.asarray(d)))

    reconstructed_images = []
    original_images = []
    preds = []
    for img in images:
        img = img.to(device)
        tmp_output, tmp_preds = model(torch.unsqueeze(img, 0))
        reconstructed_images.append(from_tensor_to_numpy(tmp_output))
        original_images.append(from_tensor_to_numpy(img))
        preds.append(int(np.argmax(tmp_preds.view(-1).cpu().detach().numpy())))

    return original_images, reconstructed_images, preds

def test(model, dataloader, device):
    model.eval()
    nb_batches = len(dataloader)
    epoch_loss = 0.0
    accuracy = 0.0


    # train on the entire dataset once
    for batch_idx, (inputs, scores, _, labels) in enumerate(dataloader):
        # progress = '\b' * len(progress) + 'Progress > {:3d}'.format(int(100 * (batch_idx + 1) / nb_batches))
        # print(progress, end='')
        # sys.stdout.flush()

        # if args.use_cuda:
        #     inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

        # compute predictions
        tot_loss = 0
        batch_acc = 0.0
        for i in range(len(inputs)):
            tmp_input = inputs[i].to(device)
            tmp_output = 0
            tmp_output, tmp_preds = model(torch.unsqueeze(tmp_input, 0))
            tmp_loss = mse_loss(tmp_input, tmp_output)
            # tmp_loss.backward()
            tot_loss += tmp_loss
            batch_acc += 1.0 if int(np.argmax(tmp_preds.view(-1).cpu().detach().numpy())) == labels[i] else 0.0

        loss = float(tot_loss) / len(inputs)
        batch_acc = batch_acc / len(inputs)
        epoch_loss += loss / nb_batches
        accuracy += batch_acc / nb_batches

    return epoch_loss, accuracy


def train(model, dataloader, optimizer, device, scheduler, save_memory=False):
    # TODO parametrize loss weights
    model.train()
    nb_batches = len(dataloader)
    progress = ''

    optimizer.zero_grad()

    avg_batch_time = 0

    classification_loss = torch.nn.CrossEntropyLoss().to(device)
    # train on the entire dataset once
    for batch_idx, (inputs, scores, _, labels) in enumerate(dataloader):
        scheduler.batch_step()
        optimizer.zero_grad()
        batch_start_time = time.time()
        # progress = '\b' * len(progress) + 'Progress > {:3d} avg_time: '.format(int(100 * (batch_idx + 1) / nb_batches))
        # print(progress, end='')
        # sys.stdout.flush()

        # if args.use_cuda:
        #     inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

        # compute predictions
        tot_loss = 0
        for i in range(len(inputs)):
            tmp_input = inputs[i].to(device)
            tmp_output, tmp_preds = model(torch.unsqueeze(tmp_input, 0))
            tmp_loss = mse_loss(tmp_input, tmp_output) \
                      + 1.0 * classification_loss(tmp_preds, torch.LongTensor([labels[i]]).to(device))
            if save_memory:
                tmp_loss.backward()
            else:
                tot_loss = tot_loss + tmp_loss
        tmp_loss = 0

        # optimization step
        if not save_memory:
            loss = tot_loss / len(inputs)
            loss.backward()
        optimizer.step()

        avg_batch_time = (avg_batch_time * batch_idx + (time.time() - batch_start_time)) / (batch_idx + 1)
        # print(avg_batch_time)

    # compute epoch statistics
    tot_loss = 0
    loss = 0
    epoch_loss, acc = test(model, dataloader, device)

    return epoch_loss, acc

# parameters
num_epochs = 30
n_imgs_preview = 20

device = torch.device("cuda:0") #cpu

if __name__ == "__main__":
    seed = 16052018
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # create model
    model = DTDAutoEncoder()
    model.to(device)
    model.train()
    for param in model.parameters():
        model.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) # TODO optimizer and lr params
    scheduler = CyclicLR(optimizer) # TODO fai scegliere in parametri, mettici quella che dimezza quando raggiunge plateau
    # TODO anzi, mettici la roba che vede con quanta confidenza dice che non migliora più

    # load data
    dtd_train_ds = TextureDataset("ved47_features.mat", "dtd")
    dtd_val_ds = dtd_train_ds.split_dataset(0.1)
    dtd_train_dl = DataLoader(dtd_train_ds, batch_size=4, collate_fn=variable_size_input_collate_fn, shuffle=True)
    dtd_val_dl = DataLoader(dtd_val_ds, batch_size=1, collate_fn=variable_size_input_collate_fn, shuffle=True)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    now = datetime.datetime.now()

    model_foldername = "autoencoder_savestates/" + str(now.year) + ("%02d" % now.month) + ("%02d" % now.day) \
                       + ("_%02d" % now.hour) + ("%02d" % now.minute) + ("%02d" % now.second) + "/"


    # main cycle
    for epoch in range(num_epochs):
        print("Epoch %d / %d" % (epoch+1 , num_epochs))
        #train
        train_loss, train_acc = train(model, dtd_train_dl, optimizer, device, scheduler, save_memory=True)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print("Training MSE loss: %f" % train_losses[-1])
        print("Training Accuracy: %2.3f" % (train_accs[-1] * 100.0))
        #validation
        val_loss, val_acc = test(model, dtd_val_dl, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print("Validation MSE loss: %f" % val_losses[-1])
        print("Validation Accuracy: %2.3f" % (val_accs[-1] * 100.0))
        #save
        if (epoch + 1) % 5 == 0:
            os.makedirs(model_foldername, exist_ok=True)
            # TODO also save loss history, iteration number, implement load from checkpoint
            base_name = model_foldername + ("epoch_%03d" % epoch) + ("_mse_%1.3f" % val_losses[-1])\
                        + ("_acc_%1.3f" % val_accs[-1])
            os.makedirs(base_name + "/", exist_ok=True)
            torch.save(model, base_name + ".pth")
            or_imgs, rec_imgs, preds = inference(model, dtd_val_dl, device, n_subset=n_imgs_preview)
            for i in range(len(or_imgs)):
                conc_img = np.concatenate((or_imgs[i], rec_imgs[i]), axis=1)
                scipy.misc.toimage(conc_img, cmin=0.0, cmax=1.0).save(base_name + "/" + ("%03d" % i) + ".png")

    #test
