import torch


class MochiPrepareSigmasNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"tooltip": "Override sigma schedule and steps"}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "process"
    CATEGORY = "MochiEdit"

    def process(self, sigmas):
        sigmas = sigmas.tolist()
        if sigmas[-1] != 0.0:
            sigmas = [*sigmas, 0.0]

        return (torch.Tensor(sigmas),)
