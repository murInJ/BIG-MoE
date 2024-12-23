import math
from math import sqrt
import einx
import torch
import torch.nn.functional as F
import torchvision
from einops import einsum
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import Module


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Conv2d_cd(nn.Module):
    """
    Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing'
    By Zitong Yu & Zhuo Su, 2019
    https://arxiv.org/pdf/2003.04092v1.pdf

    @inproceedings{yu2020searching,
        title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
        author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
        booktitle= {CVPR},
        year = {2020}
    }
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

class PEER(Module):
    """
    @inproceedings{He2024MixtureOA,
    title   = {Mixture of A Million Experts},
    author  = {Xu Owen He},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:271038610}
    }
    """

    def __init__(
        self,
        dim,
        *,
        gate_dim=None,
        heads = 8,                       # tested up to 32 - (hk = heads * num_experts_per_head (16))
        num_experts = 1_000_000,         # he chose 1 million
        num_experts_per_head = 16,       # he settled on 16, but was 32 in PKM paper
        activation = nn.GELU,
        dim_key = None,
        product_key_topk = None,
        separate_embed_per_head = False, # @smerky notes that heads may retrieve same redundant neurons. this setting would allow for separate embeds per head and prevent that
        pre_rmsnorm = False,
        dropout = 0.
    ):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - 2 for product key
        k - number of keys
        """

        super().__init__()
        if gate_dim is None:
            self.gate_dim = dim
        else:
            self.gate_dim = gate_dim

        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.norm_gate = RMSNorm(self.gate_dim) if pre_rmsnorm else nn.Identity()

        # whether to do separate embedding per head

        num_expert_sets = 1 if not separate_embed_per_head else heads

        self.heads = heads
        self.separate_embed_per_head = separate_embed_per_head
        self.num_experts = num_experts

        # experts that will form the mlp project in / out weights

        self.weight_down_embed = nn.Embedding(num_experts * num_expert_sets, dim)
        self.weight_up_embed = nn.Embedding(num_experts * num_expert_sets, dim)

        # activation function, defaults to gelu

        self.activation = activation()

        # queries and keys for product-key

        assert sqrt(num_experts).is_integer(), '`num_experts` needs to be a square'
        assert (dim % 2) == 0, 'feature dimension should be divisible by 2'

        dim_key = default(dim_key, self.gate_dim // 2)
        self.num_keys = int(sqrt(num_experts))

        self.to_queries = nn.Sequential(
            nn.Linear(self.gate_dim, dim_key * heads * 2, bias = False),
            Rearrange('b n (p h d) -> p b n h d', p = 2, h = heads)
        )

        self.product_key_topk = default(product_key_topk, num_experts_per_head)
        self.num_experts_per_head = num_experts_per_head

        self.keys = nn.Parameter(torch.zeros(heads, self.num_keys, 2, dim_key))
        nn.init.normal_(self.keys, std = 0.02)

        # dropout

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        x_in_gate=None,
    ):

        x = self.norm(x)
        if x_in_gate is None:
            gate = self.norm(x)
        else:
            gate = self.norm_gate(x_in_gate)

        # queries

        queries = self.to_queries(gate)

        # first get similarity with keys

        sim = einsum(queries, self.keys, 'p b n h d, h k p d -> p b n h k')

        # product key logic

        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.product_key_topk, dim = -1)


        all_scores = einx.add('... i, ... j -> ... (i j)', scores_x, scores_y)
        all_indices = einx.add('... i, ... j -> ... (i j)', indices_x * self.num_keys, indices_y)

        scores, pk_indices = all_scores.topk(self.num_experts_per_head, dim = -1)

        indices = all_indices.gather(-1, pk_indices)

        # if separate embeds per head, add appropriate offsets per head

        if self.separate_embed_per_head:
            head_expert_offsets = torch.arange(self.heads, device = x.device) * self.num_experts
            indices = einx.add('b n h k, h -> b n h k', indices, head_expert_offsets)

        # build the weight matrices for projecting in and out
        # basically the experts are the gathered parameters for an MLP

        weights_down = self.weight_down_embed(indices)
        weights_up = self.weight_up_embed(indices)

        # below is basically Algorithm 1 in paper

        x = einsum(x, weights_down, 'b n d, b n h k d -> b n h k')

        x = self.activation(x)
        x = self.dropout(x)

        x = x * scores.softmax(dim = -1)

        x = einsum(x, weights_up, 'b n h k, b n h k d -> b n d')


        return x

class Modallity_Related_VP(nn.Module):
    def __init__(self,prompt_size=(3,224,224),modality_num=3,theta=0.7):
        super().__init__()
        self.prompt_size = prompt_size
        self.conv_d = Conv2d_cd(in_channels=3*modality_num, out_channels=3, kernel_size=3, stride=1,
                                      padding=1, theta=theta)
        self.modality_prompt = nn.Parameter(torch.zeros((2 ** modality_num,)+self.prompt_size))
        self.P_s = nn.Parameter(torch.zeros(prompt_size))

        self.act = QuickGELU()
        self.conv_proj = nn.Conv2d(in_channels=3*3, out_channels=64, kernel_size=16, stride=16)
        # nn.init.xavier_uniform_(self.conv_d.weight)
        # nn.init.zeros_(self.conv_d.bias)
    def forward(self,x_RGB,x_IR,x_DEPTH,modality_code):
        b,c,fh,fw = x_RGB.size()
        x_MIX = torch.cat([x_RGB,x_IR,x_DEPTH],dim=1) # b,m*c,14,14
        P_d = self.act(self.conv_d(x_MIX)) # b,c,14,14

        mts = []
        for batch_index in modality_code:
            mts.append(self.modality_prompt[batch_index])
        P_m = torch.stack(mts,dim=0)# b,3,14,14

        P_s = self.P_s.expand(b,-1,-1,-1)

        P_combine = torch.cat([P_d,P_m,P_s],dim=1) #b,9,14,14

        P_combine = self.conv_proj(P_combine).flatten(2).transpose(1, 2)

        return P_combine

class VP_Block(nn.Module):
    def __init__(self,gate_dim=64,theta=0.7):
        super().__init__()
        self.conv_VP = Conv2d_cd(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                padding=1, theta=theta)
        self.conv_gate = Conv2d_cd(in_channels=gate_dim, out_channels=64, kernel_size=3, stride=1,
                                 padding=1, theta=theta)
        self.act = QuickGELU()
        self.attn = eca_layer(64+64)
        self.conv_compose = Conv2d_cd(in_channels=64+64, out_channels=64, kernel_size=3, stride=1,
                                 padding=1, theta=theta)
        self.drop = nn.Dropout(p=0.1)
        # nn.init.xavier_uniform_(self.conv_VP.weight)
        # nn.init.zeros_(self.conv_VP.bias)
        # nn.init.xavier_uniform_(self.conv_gate.weight)
        # nn.init.zeros_(self.conv_gate.bias)
        # nn.init.xavier_uniform_(self.conv_compose.weight)
        # nn.init.zeros_(self.conv_compose.bias)

    def forward(self,VP,gate):# b,l,64  b,l,gate_dim
        b,l,d = gate.size()
        gate = gate.reshape(b,14,14,d).transpose(1,3) #b,d,14,14
        gate = self.act(self.conv_gate(gate))
        VP = VP.reshape(b,14,14,64).transpose(1,3)#b,64,14,14
        VP = self.act(self.conv_VP(VP))
        VP = torch.cat([VP,gate],dim=1)
        VP = self.attn(VP)
        VP = self.drop(VP)
        VP = self.act(self.conv_compose(VP))
        VP = VP.reshape(b,64,14*14).transpose(1,2)
        return VP

class PEER_Adaptor_VP(nn.Module):
    def __init__(self,down_dim=64,dim=768,modality_num=3,num_experts=40*40):
        super().__init__()
        self.G_down = nn.Linear(modality_num*down_dim, down_dim)
        self.C_down = nn.Linear(dim,down_dim)
        self.adaptor = PEER(dim=down_dim*modality_num,num_experts=num_experts,num_experts_per_head=2,pre_rmsnorm=True,gate_dim=down_dim*2,heads=2)
        self.conv_up = nn.Linear(down_dim, dim)
        self.act = QuickGELU()
        self.modality_num = modality_num
        self.down_dim = down_dim
        self.VP_block = VP_Block(gate_dim=down_dim*2)
        self.conv_VP = nn.Linear(64,down_dim)
        self.drop = nn.Dropout(p=0.1)

        # nn.init.xavier_uniform_(self.G_down.weight)
        # nn.init.zeros_(self.G_down.bias)
        # nn.init.xavier_uniform_(self.C_down.weight)
        # nn.init.zeros_(self.C_down.bias)
        # nn.init.zeros_(self.conv_up.weight)
        # nn.init.zeros_(self.conv_up.bias)
    def forward(self,x,VP):
        b,l,d = x.size() # b,m*l+1,d

        cls = x[:, :1]  # b,1,d

        C = x[:, 1:14*14*self.modality_num+1]  # b,3l,d
        C = self.act(self.C_down(C))  # b,3l,down_dim
        C = C.reshape(b, self.modality_num, -1, self.down_dim)  # [b,m,l,down_dim]
        C = C.permute(0, 2, 1, 3)  # [b,l,m,down_dim]
        C = C.reshape(b, -1, self.down_dim * self.modality_num)  # [b,l,m*down_dim]

        # gate
        G = self.act(self.G_down(C))  # b,l,down_dim
        VP_G = self.act(self.conv_VP(VP))#b,l,down_dim
        G = torch.cat([G,VP_G],dim=-1)
        VP = self.VP_block(VP,G)

        # adaptor
        C= self.adaptor(C, x_in_gate=G)  # b,l,m*down_dim
        C = C.reshape(b, -1, self.modality_num, self.down_dim)  # b,l,m,down_dim
        C = C.permute(0, 2, 1, 3)  # [b,m,l,down_dim]
        C = C.reshape(b, -1, self.down_dim)  # b,m*l,down_dim
        C = self.drop(C)
        C = self.act(self.conv_up(C))  # b,m*l,d

        Z = torch.cat([cls, C], dim=1)
        x = x + Z
        return x,VP

class BIG_MoE(nn.Module):
    def __init__(self):
        super(BIG_MoE, self).__init__()
        self.modality_num = 3
        self.num_encoders = 12
        self.dim = 768

        # Load imagenet pretrained ViT Base 16 and freeze all parameters first
        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters(): p.requires_grad = True

        # extract encoder alone and discard CNN (patchify + linear projection) feature extractor, classifer head
        # Refer Encoder() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html

        self.conv_proj = vit_b_16.conv_proj
        for p in self.conv_proj.parameters(): p.requires_grad = False

        vit = vit_b_16.encoder
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        self.seq_length = 14 * 14

        self.pos_embedding = nn.Parameter(torch.empty(1, self.seq_length*self.modality_num+1, self.dim).normal_(std=0.02))

        # start building ViT encoder layers
        self.down_dim = 8
        self.ViT_Encoder = nn.ModuleList()
        self.adaptors = nn.ModuleList()

        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters(): p.requires_grad = False
            self.ViT_Encoder.append(vit.layers[i])
            self.adaptors.append(PEER_Adaptor_VP(down_dim=self.down_dim,dim=self.dim,modality_num=self.modality_num))


        self.fc = nn.Sequential(
            nn.Linear(self.dim , 2),
        )
        self.act = QuickGELU()

        self.VP_learner = Modallity_Related_VP()


    def forward(self, x_RGB, x_IR, x_DEPTH,mask_prob=0.3):
        b, c, fh, fw = x_RGB.size()

        coded = torch.zeros(b, dtype=torch.int64, device=x_RGB.device)
        if  self.training and self.modality_num > 1:
            for batch in range(b):
                drop_mask = torch.bernoulli(torch.full((self.modality_num,), mask_prob))
                for modality, drop in enumerate(drop_mask):
                    if drop:
                        if modality == 0:
                            x_RGB[batch] = torch.zeros_like(x_RGB[batch], device=x_RGB.device)
                        elif modality == 1:
                            x_IR[batch] = torch.zeros_like(x_IR[batch], device=x_IR.device)
                        elif modality == 2:
                            x_DEPTH[batch] = torch.zeros_like(x_DEPTH[batch], device=x_DEPTH.device)

                        coded[batch] |= (1 << modality)

        VP = self.VP_learner(x_RGB,x_IR,x_DEPTH,coded)


        x_RGB = self.conv_proj(x_RGB).flatten(2).transpose(1, 2)  # b, gh*gw, d
        x_IR = self.conv_proj(x_IR).flatten(2).transpose(1, 2)  # b, gh*gw, d
        x_DEPTH = self.conv_proj(x_DEPTH).flatten(2).transpose(1, 2)  # b, gh*gw, d
        x_MIX = torch.cat([self.class_token.expand(b, -1, -1), x_RGB, x_IR, x_DEPTH], dim=1) + self.pos_embedding

        for i, encoder in enumerate(self.ViT_Encoder.children()):
            # SA
            x_MIX = encoder.ln_1(x_MIX)  # b,3l+1,d
            xx_MIX, W_MIX = encoder.self_attention(x_MIX, x_MIX, x_MIX, need_weights=True)  # [b,3l+1,d],[b,3l,3l]
            xx_MIX = encoder.dropout(xx_MIX)
            x_MIX = encoder.ln_2(x_MIX + xx_MIX)
            self.attn_map = W_MIX
            # pretrained
            F_MIX = encoder.mlp(x_MIX)  # b,3l+1,d

            # adaptor
            Z_MIX,VP = self.adaptors[i](x_MIX,VP)  # b,l,m*down_dim

            x_MIX = x_MIX + F_MIX + Z_MIX

        MIX_feature = x_MIX[:, 1:]

        self.feature_output = MIX_feature.view(b, -1)

        logits = self.fc(x_MIX[:, 0])  # b,num_classes

        logits = {
            'MIX':logits
        }
        # features = (features[:,1:] @ self.proj).mean(dim=1)
        # average predictions for every token
        return logits

    def extra_training_loss_backward(self,spoof_label=None):
        if hasattr(self, 'total_aux_loss'):
            self.total_aux_loss.backward(retain_graph=True)

    def get_output_feature(self):
        return self.feature_output
    def get_attn_map(self):
        return self.attn_map