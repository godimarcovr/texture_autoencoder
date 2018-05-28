from torch import nn
import torch
from torch.autograd import Variable

class DTDAutoEncoder(nn.Module):
    def __init__(self):
        super(DTDAutoEncoder, self).__init__()
        self.encoder = nn.ModuleList([
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # b, 64, w, h
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # b, 64, w, h
            nn.ReLU(True),
            nn.MaxPool2d(2),  # b, 64, w/2, h/2
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # b, 64, w/2, h/2
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # b, 64, w/2, h/2
            nn.ReLU(True),
            nn.MaxPool2d(2),  # b, 64, w/4, h/4
        ])
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # b, 64, w/4, h/4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # b, 64, w/4, h/4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0),  # b, 32, w/2, h/2
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # b, 64, w/2, h/2
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=0),  # b, 3, w, h
            nn.ReLU(True)
        ])

        self.classifier = nn.ModuleList([
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 47)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("INPUT")
        # print(x.size())
        # print("ENCODER")
        featuremaps_sizes = [x.size()]
        for m in self.encoder:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                # print(x.size())
                featuremaps_sizes.append(x.size())

        # print("CLASSIFIER")
        preds = nn.AvgPool2d((x.size(2), x.size(3)))(x).view(1, -1)
        for m in self.classifier:
            preds = m(preds)

        # print("DECODER")
        for m in self.decoder:
            if isinstance(m, nn.ConvTranspose2d):
                x = m(x, output_size=featuremaps_sizes[-1])
                featuremaps_sizes = featuremaps_sizes[:-1]
                # print(x.size())
            else:
                x = m(x)
        return x, preds