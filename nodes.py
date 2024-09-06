import torch
import numpy as np

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

class CropHalfMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "part": (["lower", "upper"],),  # 可选项定义方式
            }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)

    FUNCTION = "get_half"

    def get_half(self, mask, part):
        # 将 mask 确保为 numpy 数组，避免使用 PyTorch 张量进行计算
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()  # 如果输入是 PyTorch 张量，则转换为 numpy 数组

        # 确保 mask 是一个 3D numpy 数组: (B, H, W)
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        height, width = mask.shape[-2], mask.shape[-1]

        # 确定实际 mask 区域的上下边界
        combined_mask = np.any(mask, axis=0)  # 使用 numpy.any() 得到 (H, W) 的布尔数组
        non_zero_rows = np.where(np.any(combined_mask, axis=1))[0]  # 找出非零行的索引

        # 如果没有非零值区域，返回一个全零的 mask
        if len(non_zero_rows) == 0:
            return (torch.zeros_like(torch.from_numpy(mask).float()),)  # 返回全零张量

        # 计算实际 mask 区域的上边界和下边界
        upper_bound = non_zero_rows[0]  # 实际 mask 的上边界
        lower_bound = non_zero_rows[-1] + 1  # 实际 mask 的下边界

        # 根据实际 mask 区域计算中间行的位置
        middle_row = (upper_bound + lower_bound) // 2

        # 创建一个与原始 mask 大小相同的全零数组
        new_mask = np.zeros_like(mask)

        if part == "lower":
            # 提取实际 mask 下半部分并粘贴到新 mask 的相应位置
            lower_half = mask[:, middle_row:lower_bound, :]
            new_mask[:, middle_row:lower_bound, :] = lower_half
        elif part == "upper":
            # 提取实际 mask 上半部分并粘贴到新 mask 的相应位置
            upper_half = mask[:, upper_bound:middle_row, :]
            new_mask[:, upper_bound:middle_row, :] = upper_half

        # 转换为 PyTorch 张量
        new_mask_tensor = torch.from_numpy(new_mask).float()

        # 返回结果，确保是单元素元组
        return (new_mask_tensor,)


class CropMaskByRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "part": (["lower", "upper"],),  # 可选项定义方式
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # 用于控制区域占比
            }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)

    FUNCTION = "crop_by_ratio"

    def crop_by_ratio(self, mask, part, ratio):
        # 将 mask 确保为 numpy 数组，避免使用 PyTorch 张量进行计算
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()  # 如果输入是 PyTorch 张量，则转换为 numpy 数组

        # 确保 mask 是一个 3D numpy 数组: (B, H, W)
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        height, width = mask.shape[-2], mask.shape[-1]

        # 确定实际 mask 区域的上下边界
        combined_mask = np.any(mask, axis=0)  # 使用 numpy.any() 得到 (H, W) 的布尔数组
        non_zero_rows = np.where(np.any(combined_mask, axis=1))[0]  # 找出非零行的索引

        # 如果没有非零值区域，返回一个全零的 mask
        if len(non_zero_rows) == 0:
            return (torch.zeros_like(torch.from_numpy(mask).float()),)  # 返回全零张量

        # 计算实际 mask 区域的上边界和下边界
        upper_bound = non_zero_rows[0]  # 实际 mask 的上边界
        lower_bound = non_zero_rows[-1] + 1  # 实际 mask 的下边界
        actual_height = lower_bound - upper_bound

        # 计算分割位置，基于实际区域的高度和给定的比例
        split_row = upper_bound + int(actual_height * ratio)

        # 创建一个与原始 mask 大小相同的全零数组
        new_mask = np.zeros_like(mask)

        if part == "lower":
            # 提取实际 mask 下部分并粘贴到新 mask 的相应位置
            lower_half = mask[:, split_row:lower_bound, :]
            new_mask[:, split_row:lower_bound, :] = lower_half
        elif part == "upper":
            # 提取实际 mask 上部分并粘贴到新 mask 的相应位置
            upper_half = mask[:, upper_bound:split_row, :]
            new_mask[:, upper_bound:split_row, :] = upper_half

        # 转换为 PyTorch 张量
        new_mask_tensor = torch.from_numpy(new_mask).float()

        # 返回结果，确保是单元素元组
        return (new_mask_tensor,)

NODE_CLASS_MAPPINGS = {
    "union-MaskCombine": MaskComposite,
    "union-CropHalfMask": CropHalfMask,
    "union-CropMaskByRatio": CropMaskByRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskCombine": "union-MaskCombine",
    "CropHalfMask": "union-CropHalfMask",
    "CropMaskByRatio": "union-CropMaskByRatio"
}