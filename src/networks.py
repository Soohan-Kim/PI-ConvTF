import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Implementation of Self-Attention Memory Module
class SelfAttentionMemory(nn.Module):
    def __init__(self, input_channels, inter_channels):
        super(SelfAttentionMemory, self).__init__()
        
        self.input_channels = input_channels
        self.inter_channels = inter_channels
        
        self.Wq = nn.Conv2d(in_channels=input_channels, out_channels=inter_channels, kernel_size=1)
        self.Whk = nn.Conv2d(in_channels=input_channels, out_channels=inter_channels, kernel_size=1)
        self.Whv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmk = nn.Conv2d(in_channels=input_channels, out_channels=inter_channels, kernel_size=1)
        self.Wmv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)

        self.Wz = nn.Conv2d(in_channels=2*input_channels, out_channels=input_channels, kernel_size=1)
        
        self.Wmzo = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmho = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmzg = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmhg = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmzi = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmhi = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)

    def forward(self, x):
        ht, mt_1 = x
        feature_map_size = ht.size(dim=-1)

        Q = self.Wq(ht)
        Kh, Vh, Km, Vm = self.Whk(ht), self.Whv(ht), self.Wmk(mt_1), self.Wmv(mt_1)

        Q = torch.flatten(Q, start_dim=-2)
        QT = torch.transpose(Q, -2, -1)
        Kh, Vh, Km, Vm = torch.flatten(Kh, start_dim=-2), torch.flatten(Vh, start_dim=-2), torch.flatten(Km, start_dim=-2), torch.flatten(Vm, start_dim=-2)

        Ah, Am = F.softmax(torch.matmul(QT, Kh), dim=-1), F.softmax(torch.matmul(QT, Km), dim=-1)
        Zh, Zm = torch.matmul(Vh, torch.transpose(Ah, -2, -1)), torch.matmul(Vm, torch.transpose(Am, -2, -1))

        Z = self.Wz(torch.cat((Zh, Zm), dim=-2).view(ht.size(dim=0), ht.size(dim=1)*2, feature_map_size, feature_map_size))

        it = torch.sigmoid(self.Wmzi(Z) + self.Wmhi(ht))
        gt = torch.tanh(self.Wmzg(Z) + self.Wmhg(ht))
        Mt = (1-it) * mt_1 + it * gt

        ot = torch.sigmoid(self.Wmzo(Z) + self.Wmho(ht))
        Ht = ot * Mt

        return Ht, Mt

# Implementation of a SA-ConvLSTM Cell w/ Self-Attention Memory Module
## If inter_channels is not given (None), it reduces to a vanilla ConvLSTM Cell
class SAConvLSTMCell(nn.Module):
    def __init__(self, input_channels, feature_channels, inter_channels, kernel_size, stride, padding, device):
        super(SAConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.inter_channels = inter_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        self.Wxi = nn.Conv2d(in_channels=input_channels, out_channels=feature_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Whi = nn.Conv2d(in_channels=feature_channels, out_channels=feature_channels, kernel_size=1)
        self.Wxf = nn.Conv2d(in_channels=input_channels, out_channels=feature_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Whf = nn.Conv2d(in_channels=feature_channels, out_channels=feature_channels, kernel_size=1)
        self.Wxg = nn.Conv2d(in_channels=input_channels, out_channels=feature_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Whg = nn.Conv2d(in_channels=feature_channels, out_channels=feature_channels, kernel_size=1)
        self.Wxo = nn.Conv2d(in_channels=input_channels, out_channels=feature_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Who = nn.Conv2d(in_channels=feature_channels, out_channels=feature_channels, kernel_size=1)

        if self.inter_channels != 'None':
            self.SAM = SelfAttentionMemory(input_channels=feature_channels, inter_channels=inter_channels)

    def forward(self, x, others):
        ct_1, ht_1, mt_1 = others

        it = torch.sigmoid(self.Wxi(x) + self.Whi(ht_1))
        ft = torch.sigmoid(self.Wxf(x) + self.Whf(ht_1))
        gt = torch.tanh(self.Wxg(x) + self.Whg(ht_1))

        Ct = ft * ct_1 + it * gt
        
        ot = torch.sigmoid(self.Wxo(x) + self.Who(ht_1))
        Ht = ot * torch.tanh(Ct)

        if self.inter_channels != 'None':
            Ht, Mt = self.SAM((Ht, mt_1))
        else:
            Mt = mt_1

        return Ct, Ht, Mt

    def init_hidden(self, batch_size, image_size):
        return (torch.zeros(batch_size, self.feature_channels, image_size, image_size, device=self.device), # device=self.device
                torch.zeros(batch_size, self.feature_channels, image_size, image_size, device=self.device), # device=self.device
                torch.zeros(batch_size, self.feature_channels, image_size, image_size, device=self.device)) # device=self.device

# Implementation of SA-ConvLSTM Network
class SAConvLSTM(nn.Module):
    def __init__(self, input_channels, feature_channels, inter_channels, kernel_size, stride, padding, device, last_conv, num_layers):
        '''
        inputs
            -input_channels: input tensor channel dimension (int)
            -feature_channels: list of filter numbers per layer (list of ints)
            -inter_channels: list of Query, Key channel dimensions per layer (list of ints)
            -kernel_size: list of kernel_sizes to apply convolution operations on cell inputs per layer (list of ints)
            -stride: list of stride sizes to apply convolution operations on cell inputs per layer (list of ints)
            -padding: list of padding sizes to apply convolution operations on cell inputs per layer (list of ints)
            -device: device to train model on ('cuda' or 'cpu')
            -last_conv: [last conv kernel size, last conv stride, last conv padding] (list of ints)
            -num_layers: number of SAConvLSTM layers (int)
        '''
        super(SAConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.inter_channels = inter_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        self.num_layers = num_layers
        last_kernel, last_stride, last_padding = last_conv[0], last_conv[1], last_conv[2]

        SACellList = []
        for i in range(num_layers):
            in_chan = input_channels if i == 0 else feature_channels[i-1]
            SACellList.append(
                SAConvLSTMCell(in_chan, feature_channels[i], inter_channels[i], kernel_size[i], stride[i], padding[i], device)
            )

        self.cell_list = nn.ModuleList(SACellList)
        self.final_convout = nn.Conv2d(in_channels=feature_channels[num_layers-1], out_channels=1, kernel_size=last_kernel, stride=last_stride, padding=last_padding)

    def forward(self, x):
        hidden_sizes = [math.floor((x.size(dim=-1) + 2*self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)]
        for i in range(self.num_layers-1):
            hidden_sizes.append(math.floor((hidden_sizes[i] + 2*self.padding[i+1] - self.kernel_size[i+1]) / self.stride[i+1] + 1))
        hidden_states = self._init_hidden(x.size(dim=0), hidden_sizes)

        layer_output_list = []
        seq_len = x.size(dim=1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            c, h, m = hidden_states[layer_idx]
            output_per_layer = []
            for t in range(seq_len):
                c, h, m = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :], others=(c, h, m))
                output_per_layer.append(h)

            layer_output = torch.stack(output_per_layer, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)

        out = layer_output_list[-1][:, -1, :, :, :]
        out = self.final_convout(out)
        
        return out

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size[i]))

        return init_states

# Helper function for positional encoding for ConvTF, PI-ConvTF
def PositionalEncoding(input, t):
    d_model = input.size(dim=-3)

    for d in range(d_model):
        if d % 2 == 0:
            input[:, d, :, :] += math.sin(t / (10000)**(d / d_model))
        else:
            input[:, d, :, :] += math.cos(t / (10000)**((d-1) / d_model))

    return input

# Implementation of ConvTF Feature Embedding Layer
class FeatureEmbedding(nn.Module):
    def __init__(self, more_inputs, seq_len, channels_list, kernels_list, stride_list, paddings_list):
        super(FeatureEmbedding, self).__init__()

        self.more_inputs = more_inputs
        self.seq_len = seq_len
        self.channels = channels_list
        self.kernels = kernels_list
        self.strides = stride_list
        self.paddings = paddings_list

        feature_embeds = []
        for i in range(4):
            if i == 0:
                self.in_channels = 5 if more_inputs else 1
            else:
                self.in_channels = self.channels[i-1]

            feature_embeds.append(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.channels[i], kernel_size=self.kernels[i], stride=self.strides[i], padding=self.paddings[i])
            )
            feature_embeds.append(
                nn.LeakyReLU()
            )

        self.feature_embedding = nn.Sequential(*nn.ModuleList(feature_embeds))

    def forward(self, x):
        out = []
        for t in range(self.seq_len):
            cur_x = x[:, t, :, :, :]
            cur_out = PositionalEncoding(self.feature_embedding(cur_x), t)
            out.append(cur_out)

        out = torch.stack(out, dim=1)

        return out

# Implementation of the ConvTF Multi-Convolution Attention Layer
class MultiConvAttn(nn.Module):
    def __init__(self, num_heads, d_model, other_query=False):
        super(MultiConvAttn, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.other_query = other_query

        assert self.d_model % self.num_heads == 0

        self.conv_attn_layers = nn.ModuleList()
        for i in range(self.num_heads):

            self.conv_attn_layers += [
                nn.Conv2d(in_channels=d_model, out_channels=d_model//num_heads, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=d_model, out_channels=d_model//num_heads, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=2*d_model//num_heads, out_channels=1, kernel_size=3, stride=1, padding=1)
            ]

    def forward(self, x, input_for_query=None):
        out_per_head = []
        for i in range(self.num_heads):
            out_per_seq = []

            if self.other_query:
                assert input_for_query is not None
                tensor_for_query = input_for_query

            else: 
                tensor_for_query = x

            for j in range(tensor_for_query.size(dim=1)):
                query_j = self.conv_attn_layers[3*i](tensor_for_query[:, j, :, :, :])

                values, hidden_maps = [], []

                for k in range(x.size(dim=1)):
                    key_k = self.conv_attn_layers[3*i+1](x[:, k, :, :, :])
                    value_k = key_k
                    values.append(value_k)

                    q_k_concat = torch.cat([query_j, key_k], dim=-3)

                    H_k = self.conv_attn_layers[3*i+2](q_k_concat)

                    hidden_maps.append(H_k)

                H_j = F.softmax(torch.stack(hidden_maps, dim=1), dim=1)

                V_j = torch.stack(values, dim=1)

                V_i = torch.sum(H_j * V_j, dim=1)

                out_per_seq.append(V_i)

            out_seq = torch.stack(out_per_seq, dim=1)
            out_per_head.append(out_seq)

        out = torch.cat(out_per_head, dim=-3)

        return out

# Implementation of ConvTF Encoder Layer
class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, hidden_size):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.hidden_size = hidden_size

        self.multi_attention = MultiConvAttn(num_heads=self.num_heads, d_model=self.d_model)
        self.layer_norm1 = nn.LayerNorm([d_model, hidden_size, hidden_size])

        self.feed_forward = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.layer_norm2 = nn.LayerNorm([d_model, hidden_size, hidden_size])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layer_norm1(x + self.multi_attention(x))
            feed_forward_outs = []
            for t in range(x.size(dim=1)):
                cur_in = x[:, t, :, :, :]
                feed_forward_outs.append(self.layer_norm2(cur_in + self.feed_forward(cur_in)))

            x = torch.stack(feed_forward_outs, dim=1)

        return x

# Implementation of ConvTF Decoder Layer
class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, hidden_size):
        super(Decoder, self).__init__()

        self.num_layers, self.num_heads, self.d_model, self.hidden_size = num_layers, num_heads, d_model, hidden_size

        self.query_attention = MultiConvAttn(num_heads=self.num_heads, d_model=self.d_model)
        self.layer_norm1 = nn.LayerNorm([d_model, hidden_size, hidden_size])

        self.multi_attention = MultiConvAttn(num_heads=self.num_heads, d_model=self.d_model, other_query=True)
        self.layer_norm2 = nn.LayerNorm([d_model, hidden_size, hidden_size])

        self.feed_forward = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.layer_norm3 = nn.LayerNorm([d_model, hidden_size, hidden_size]) 

    def forward(self, x):
        input_for_query = x[:, -1, :, :, :]
        input_for_query = torch.unsqueeze(input_for_query, dim=1)

        for i in range(self.num_layers):
            input_for_query = self.layer_norm1(input_for_query + self.query_attention(input_for_query))

            input_for_query = self.layer_norm2(input_for_query + self.multi_attention(x, input_for_query))

            input_for_query = self.layer_norm3(input_for_query + torch.unsqueeze(self.feed_forward(torch.squeeze(input_for_query, dim=1)), dim=1))

        return input_for_query

# Implementation of ConvTF Synthetic Feed-Forward Network Layer
class SFFN(nn.Module):
    def __init__(self, sffn_config):
        super(SFFN, self).__init__()

        self.sffn_config = sffn_config

        self.sffn1 = nn.ModuleList()
        self.sffn2 = nn.ModuleList()

        for i in range(10):
            self.sffn1 += [
                nn.Conv2d(in_channels=sffn_config[i][0], out_channels=sffn_config[i][1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            ]

        self.sffn1.append(
            nn.Conv2d(in_channels=sffn_config[10][0], out_channels=sffn_config[10][1], kernel_size=1)
        )

        for i in range(18):
            self.sffn2 += [
                nn.Conv2d(in_channels=sffn_config[i+11][0], out_channels=sffn_config[i+11][1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            ]

        self.sffn2.append(
            nn.Conv2d(in_channels=sffn_config[-1][0], out_channels=sffn_config[-1][1], kernel_size=1)
        )

        self.sffn1 = nn.Sequential(*self.sffn1)
        self.sffn2 = nn.Sequential(*self.sffn2)

    def forward(self, x):
        out = self.sffn2(self.sffn1(x)) + self.sffn1(x)

        return out

# Implementation of ConvTF Network
class ConvTransformer(nn.Module):
    def __init__(self, more_inputs,
        seq_len, channels_list, 
        num_layers, num_heads, d_model, hidden_size,
        sffn_config,
        kernels_list=[3, 3, 3, 3], stride_list=[1, 1, 1, 1], paddings_list=[1, 1, 1, 1]
    ):
        '''
        inputs
            -more_inputs: flag for training with more inputs or only volatility (bool)
            -seq_len: past timestep number to use for prediction (int)
            -channels_list: number of output channels for the feature embedding layer (list of ints)
            -num_layers: number of encoder and decoder layers (int)
            -num_heads: number of attention heads to use for Multi-Convolution Attention (int)
            -d_model: number of channels of the input feature maps to the encoder layer (int)
            -hidden_size: input size or grid size (int) -> required information for layer normalization in Multi-Convolution Attention
            -sffn_config: configuration for Synthetic Feed-Forward Network layer (None or list of list of ints)
        '''
        super(ConvTransformer, self).__init__()

        self.feature_embed = FeatureEmbedding(more_inputs, seq_len, channels_list, kernels_list, stride_list, paddings_list)
        self.encoder = Encoder(num_layers, num_heads, d_model, hidden_size)
        self.decoder = Decoder(num_layers, num_heads, d_model, hidden_size)

        self.sffn_config = sffn_config
        if sffn_config is not None:
            self.sffn = SFFN(sffn_config)
        else:
            self.direct_pred = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.feature_embed(x)
        out = self.encoder(x)
        out = self.decoder(out)
        out = torch.squeeze(out, dim=1)

        if self.sffn_config is not None:
            out = self.sffn(out)
        else:
            out = self.direct_pred(out)

        out = self.leaky_relu(out)

        return out