import torch
import torch.nn as nn


def padding(input, output, k, s):
    return int( ((output-1) *s -input+k) /2)
def output(input, p, k, s):
    return int( (input + 2*p - k) / s) + 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        """
        My custom ResidualBlock

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size

        [hint]
        * See the instruction PDF for details
        * Set the bias argument to False
        """
        
        ## Define all the layers
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)
        self.relu = nn.ReLU()


    def forward(self, x):
    #    output = int( (input + 2*p - k) / s) + 1
    #    padding = int( ((output-1) *s -input+k) /2)
        residual = self.residual(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x += residual
        x = self.relu(x)
        return x



class MyResnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        """
        My custom ResNet.

        [input]
        * in_channels  : input channel number
        * num_classes  : number of classes

        [hint]
        * See the instruction PDF for details
        * Set the bias argument to False
        """
        
        ## Define all the layers
        # stride=1, padding = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.block1 = ResidualBlock(in_channels=64, out_channels=2*64, kernel_size=3, stride=2)
        self.block2= ResidualBlock(in_channels=2*64, out_channels=4*64, kernel_size=3, stride=2)
        self.block3= ResidualBlock(in_channels=4*64, out_channels=8*64, kernel_size=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8*64, num_classes)
        
    
    def forward(self, x, return_embed=False):
        """
        Forward path.

        [input]
        * x             : input data
        * return_embed  : whether return the feature map of the last conv layer or not

        [output]
        * output        : output data
        * embedding     : the feature map after the last conv layer (optional)
        
        [hint]
        * See the instruction PDF for network details
        * You want to set return_embed to True if you are dealing with CAM
        """
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if return_embed:
            embed = x
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        if return_embed:
            return x, embed
        else:
            return x        


def init_weights_kaiming(m):

    """
    Kaming initialization.

    [input]
    * m : torch.nn.Module

    [hint]
    * Refer to the course slides/recitations for more details
    * Initialize the bias term in linear layer by a small constant, e.g., 0.01
    """

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
        
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

if __name__ == "__main__":
    # set model
    net = MyResnet(in_channels=3, num_classes=10)
    net.apply(init_weights_kaiming)
    # sanity check
    input = torch.randn((64, 3, 32, 32), requires_grad=True)
    output = net(input)
    print(output.shape)
    