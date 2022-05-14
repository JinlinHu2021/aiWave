import torch
import learn_wavelet_trans
import Quant
import factorized_entropy_model
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.Wavelet_Trans = learn_wavelet_trans.Wavelet_Trans(trainable_set=True)
        # self.inver_transform = learn_wavelet_trans.Wavelet_Forward(trainable_set=True)
        self.quant = Quant.Quant()
        self.dequant = Quant.DeQuant()
        self.trans_steps = 2
        # self.scale = 10000.0
        self.coding_net0 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net1 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net2 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net3 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net4 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net5 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net6 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net7 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net8 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net9 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net10 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net11 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net12 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net13 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net14 = factorized_entropy_model.Entropy_bottleneck()
        self.coding_net15 = factorized_entropy_model.Entropy_bottleneck()
        self.scale = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x):
        # forward transform
        num_pixels = x.size()[0] * x.size()[2] * x.size()[3] * x.size()[4]
        LLL = x
        HLL_list = []
        LHL_list = []
        HHL_list = []
        LLH_list = []
        HLH_list = []
        LHH_list = []
        HHH_list = []

        for i in range(self.trans_steps):
            LLL, HLL, LHL, HHL, LLH, HLH, LHH, HHH = self.Wavelet_Trans.forward_trans(LLL)
            # HLL_list.append(self.quant(HLL, self.scale))
            # LHL_list.append(self.quant(LHL, self.scale))
            # HHL_list.append(self.quant(HHL, self.scale))
            # LLH_list.append(self.quant(LLH, self.scale))
            # HLH_list.append(self.quant(HLH, self.scale))
            # LHH_list.append(self.quant(LHH, self.scale))
            # HHH_list.append(self.quant(HHH, self.scale))

            HLL_list.append(HLL)
            LHL_list.append(LHL)
            HHL_list.append(HHL)
            LLH_list.append(LLH)
            HLH_list.append(HLH)
            LHH_list.append(LHH)
            HHH_list.append(HHH)

        LLL = self.quant(LLL, self.scale)
        likelihood0 = self.coding_net0(LLL)
        bits = torch.sum(torch.log(likelihood0)) / (-np.log(2) * num_pixels)
        LLL = self.dequant(LLL, self.scale)

        likelihood1 = self.coding_net1(HLL_list[1])
        bits = bits + torch.sum(torch.log(likelihood1)) / (-np.log(2) * num_pixels)
        HLL_list[1] = self.dequant(HLL_list[1], self.scale)

        likelihood2 = self.coding_net2(LHL_list[1])
        bits = bits + torch.sum(torch.log(likelihood2)) / (-np.log(2) * num_pixels)
        LHL_list[1] = self.dequant(LHL_list[1], self.scale)

        likelihood3 = self.coding_net3(HHL_list[1])
        bits = bits + torch.sum(torch.log(likelihood3)) / (-np.log(2) * num_pixels)
        HHL_list[1] = self.dequant(HHL_list[1], self.scale)

        likelihood4 = self.coding_net4(LLH_list[1])
        bits = bits + torch.sum(torch.log(likelihood4)) / (-np.log(2) * num_pixels)
        LLH_list[1] = self.dequant(LLH_list[1], self.scale)

        likelihood5 = self.coding_net5(HLH_list[1])
        bits = bits + torch.sum(torch.log(likelihood5)) / (-np.log(2) * num_pixels)
        HLH_list[1] = self.dequant(HLH_list[1], self.scale)

        likelihood6 = self.coding_net6(LHH_list[1])
        bits = bits + torch.sum(torch.log(likelihood6)) / (-np.log(2) * num_pixels)
        LHH_list[1] = self.dequant(LHH_list[1], self.scale)

        likelihood7 = self.coding_net7(HHH_list[1])
        bits = bits + torch.sum(torch.log(likelihood7)) / (-np.log(2) * num_pixels)
        HHH_list[1] = self.dequant(HHH_list[1], self.scale)

        likelihood8 = self.coding_net8(HLL_list[0])
        bits = bits + torch.sum(torch.log(likelihood8)) / (-np.log(2) * num_pixels)
        HLL_list[0] = self.dequant(HLL_list[0], self.scale)

        likelihood9 = self.coding_net9(LHL_list[0])
        bits = bits + torch.sum(torch.log(likelihood9)) / (-np.log(2) * num_pixels)
        LHL_list[0] = self.dequant(LHL_list[0], self.scale)

        likelihood10 = self.coding_net10(HHL_list[0])
        bits = bits + torch.sum(torch.log(likelihood10)) / (-np.log(2) * num_pixels)
        HHL_list[0] = self.dequant(HHL_list[0], self.scale)

        likelihood11 = self.coding_net11(LLH_list[0])
        bits = bits + torch.sum(torch.log(likelihood11)) / (-np.log(2) * num_pixels)
        LLH_list[0] = self.dequant(LLH_list[0], self.scale)

        likelihood12 = self.coding_net12(HLH_list[0])
        bits = bits + torch.sum(torch.log(likelihood12)) / (-np.log(2) * num_pixels)
        HLH_list[0] = self.dequant(HLH_list[0], self.scale)

        likelihood13 = self.coding_net13(LHH_list[0])
        bits = bits + torch.sum(torch.log(likelihood13)) / (-np.log(2) * num_pixels)
        LHH_list[0] = self.dequant(LHH_list[0], self.scale)

        likelihood14 = self.coding_net14(HHH_list[0])
        bits = bits + torch.sum(torch.log(likelihood14)) / (-np.log(2) * num_pixels)
        HHH_list[0] = self.dequant(HHH_list[0], self.scale)

        LLL = self.Wavelet_Trans.inverse_trans(LLL, HLL_list[1], LHL_list[1], HHL_list[1], LLH_list[1], HLH_list[1],
                                               LHH_list[1], HHH_list[1])

        LLL = self.Wavelet_Trans.inverse_trans(LLL, HLL_list[0], LHL_list[0], HHL_list[0], LLH_list[0], HLH_list[0],
                                               LHH_list[0], HHH_list[0])

        return self.mse_loss(x, LLL), bits
