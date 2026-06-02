import torch.nn as nn
from mmdet.registry import MODELS
from transformers import DINOv3ViTConfig, DINOv3ViTModel


@MODELS.register_module()
class DINOv3ViT(nn.Module):

    def __init__(self, finetuning=False, output_patches=False, layers=[2,5,8,11], layer_norm=True, *args, **kwargs):
        super(DINOv3ViT, self).__init__()
        self.finetuning = finetuning
        self.layers = layers
        self.output_patches = output_patches
        self.config = DINOv3ViTConfig()
        self.model = DINOv3ViTModel(self.config)
        self.layer_norm = layer_norm and (len(layers) + int(output_patches)) > 0
        if self.layer_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(384, eps=1e-5, elementwise_affine=True) for _ in range(len(layers) + int(output_patches))])
        if not self.finetuning:
            self._freeze()


    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, x):  # should return a tuple
        """Forward image through model

        Args:
            x (torch.Tensor): image of shape (B, C, H, W)

        Returns:
            torch.tensor: Model output
        """
        outputs = {}
        if self.layers:
            def get_hook(name):
                def hook(module, input, output):
                    outputs[name] = output
                return hook

            # register hooks
            if self.output_patches:
                self.model.embeddings.register_forward_hook(get_hook("embeddings"))
            for layer in self.layers:
                self.model.layer[layer].register_forward_hook(get_hook(layer))

        z = self.model(x)
        if len(outputs.keys()) == 0:
            outputs[11] = z.last_hidden_state

        for i, k in enumerate(outputs):
            if self.layer_norm:
                z = self.norms[i](outputs[k])
            else:
                z = outputs[k]
            # remove the [CLS] token
            z = z[:, 1:, :]
            # batch_size, num_patches, hidden_size
            B, P, D = z.shape
            h = w = int(P ** 0.5)
            z = z.permute(0, 2, 1)
            z = z.reshape(B, D, h, w)
            outputs[k] = z

        if len(outputs.keys()) > 1:
            return tuple(outputs[k] for k in outputs.keys())
        return outputs.get(next(iter(outputs.keys())))