import torch

class MaskComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "input1": ("MASK",),
                "input2": ("MASK",),
                "input3": ("MASK",),
                "input4": ("MASK",),
                "input5": ("MASK",),
                "input6": ("MASK",),
                "input7": ("MASK",),
                "input8": ("MASK",),
                "input9": ("MASK",),
                "input10": ("MASK",),
            },
        }
    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)

    FUNCTION = "combine"

    def combine(self, input1=None, input2=None, input3=None, input4=None, input5=None, input6=None, input7=None,
                input8=None, input9=None, input10=None):
        output = None

        # Iterate over all inputs and combine them
        for input_mask in [input1, input2, input3, input4, input5, input6, input7, input8, input9, input10]:
            if input_mask is not None:
                input_mask = input_mask.reshape((-1, input_mask.shape[-2], input_mask.shape[-1]))
                if output is None:
                    output = input_mask.clone()
                else:
                    left, top = (0, 0)
                    right, bottom = (min(left + input_mask.shape[-1], output.shape[-1]),
                                     min(top + input_mask.shape[-2], output.shape[-2]))
                    visible_width, visible_height = (right - left, bottom - top)
                    source_portion = input_mask[:, :visible_height, :visible_width]
                    destination_portion = output[:, top:bottom, left:right]
                    output[:, top:bottom, left:right] = destination_portion + source_portion

        output = torch.clamp(output, 0.0, 1.0)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "union-MaskCombine": MaskComposite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskCombine": "union-MaskCombine"
}