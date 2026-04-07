import torch.nn as nn
from mmdet.registry import MODELS
from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel


@MODELS.register_module()
class DINOv3ConvNext(nn.Module):

    def __init__(self, finetuning=False, layers=[0,1,2,3], *args, **kwargs):
        super(DINOv3ConvNext, self).__init__()
        self.finetuning = finetuning
        self.layers = layers
        self.config = DINOv3ConvNextConfig()
        self.model = DINOv3ConvNextModel(self.config)
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
            for i in self.layers:
                self.model.stages[i].register_forward_hook(get_hook(i))

        z = self.model(x)
        if len(outputs.keys()) == 0:
            outputs[0] = z.last_hidden_state

        if len(outputs.keys()) > 1:
            return tuple(outputs[k] for k in outputs.keys())
        return outputs.get(next(iter(outputs.keys())))