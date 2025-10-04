"""
空输入节点模块
为工作流提供默认输入值，当下级节点没有连接时使用
"""

import torch

class EmptyInputNode:
    """空输入节点 - 为工作流提供默认输入值"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": (["image", "latent", "mask", "conditioning"], {"default": "image"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "content_type": (["black", "white", "checkerboard", "gradient", "noise"], {"default": "black"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "STRING")
    RETURN_NAMES = ("image", "latent", "mask", "info")
    FUNCTION = "generate_input"
    CATEGORY = "MISLG Tools/Input"
    DESCRIPTION = "生成空输入数据，防止下级节点因无输入而报错"

    def generate_input(self, input_type, width, height, batch_size, content_type):
        info = f"生成 {input_type} 输入: {width}x{height}, {content_type}"
        
        if input_type == "image":
            image = self.create_image(width, height, batch_size, content_type)
            return (image, None, None, info)
        elif input_type == "latent":
            latent = self.create_latent(width, height, batch_size, content_type)
            return (None, latent, None, info)
        elif input_type == "mask":
            mask = self.create_mask(width, height, content_type)
            return (None, None, mask, info)
        elif input_type == "conditioning":
            latent = self.create_latent(width, height, batch_size, "zeros")
            return (None, latent, None, f"{info} (使用潜在空间作为占位符)")
        else:
            image = self.create_image(width, height, batch_size, "black")
            return (image, None, None, f"未知类型，返回默认图像: {info}")

    def create_image(self, width, height, batch_size, content_type):
        if content_type == "black":
            return torch.zeros((batch_size, height, width, 3), dtype=torch.float32)
        elif content_type == "white":
            return torch.ones((batch_size, height, width, 3), dtype=torch.float32)
        elif content_type == "checkerboard":
            return self.create_checkerboard_image(width, height, batch_size)
        elif content_type == "gradient":
            return self.create_gradient_image(width, height, batch_size)
        elif content_type == "noise":
            return torch.rand((batch_size, height, width, 3), dtype=torch.float32)
        else:
            return torch.zeros((batch_size, height, width, 3), dtype=torch.float32)

    def create_latent(self, width, height, batch_size, content_type):
        latent_height = height // 8
        latent_width = width // 8
        
        if content_type == "zeros" or content_type == "black":
            latent = torch.zeros([batch_size, 4, latent_height, latent_width])
        elif content_type == "ones" or content_type == "white":
            latent = torch.ones([batch_size, 4, latent_height, latent_width])
        elif content_type == "noise":
            latent = torch.randn([batch_size, 4, latent_height, latent_width])
        else:
            latent = torch.zeros([batch_size, 4, latent_height, latent_width])
        
        return {"samples": latent}

    def create_mask(self, width, height, content_type):
        if content_type == "black" or content_type == "zeros":
            return torch.zeros((height, width), dtype=torch.float32)
        elif content_type == "white" or content_type == "ones":
            return torch.ones((height, width), dtype=torch.float32)
        elif content_type == "checkerboard":
            return self.create_checkerboard_mask(width, height)
        elif content_type == "gradient":
            return self.create_gradient_mask(width, height)
        elif content_type == "noise":
            return torch.rand((height, width), dtype=torch.float32)
        else:
            return torch.ones((height, width), dtype=torch.float32)

    def create_checkerboard_image(self, width, height, batch_size):
        checker_size = 64
        image = torch.zeros((height, width, 3), dtype=torch.float32)
        
        for y in range(height):
            for x in range(width):
                if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                    image[y, x] = torch.tensor([0.2, 0.2, 0.2])
                else:
                    image[y, x] = torch.tensor([0.8, 0.8, 0.8])
        
        return image.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def create_gradient_image(self, width, height, batch_size):
        image = torch.zeros((height, width, 3), dtype=torch.float32)
        
        for y in range(height):
            for x in range(width):
                r = x / width
                g = y / height
                b = (x + y) / (width + height)
                image[y, x] = torch.tensor([r, g, b])
        
        return image.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def create_checkerboard_mask(self, width, height):
        checker_size = 32
        mask = torch.zeros((height, width), dtype=torch.float32)
        
        for y in range(height):
            for x in range(width):
                if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                    mask[y, x] = 0.0
                else:
                    mask[y, x] = 1.0
        
        return mask

    def create_gradient_mask(self, width, height):
        mask = torch.zeros((height, width), dtype=torch.float32)
        
        for y in range(height):
            mask[y, :] = y / height
        
        return mask

class UniversalInputNode:
    """通用输入节点 - 自动检测模式"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_type": (["image", "latent", "mask"], {"default": "image"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "content_style": (["neutral", "visible", "random"], {"default": "visible"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "STRING")
    RETURN_NAMES = ("image", "latent", "mask", "info")
    FUNCTION = "generate_input"
    CATEGORY = "MISLG Tools/Input"
    DESCRIPTION = "通用输入节点，生成各种类型的输入数据"

    def generate_input(self, data_type, width, height, content_style):
        info = f"生成 {data_type} 输入: {width}x{height}, {content_style}"
        
        if data_type == "image":
            image = self.create_content_image(width, height, content_style)
            return (image, None, None, info)
        elif data_type == "latent":
            latent = self.create_content_latent(width, height, content_style)
            return (None, latent, None, info)
        elif data_type == "mask":
            mask = self.create_content_mask(width, height, content_style)
            return (None, None, mask, info)
        else:
            image = self.create_content_image(width, height, "visible")
            return (image, None, None, f"未知类型，返回默认图像: {info}")

    def create_content_image(self, width, height, style):
        if style == "neutral":
            return torch.zeros((1, height, width, 3), dtype=torch.float32)
        elif style == "visible":
            image = torch.zeros((height, width, 3), dtype=torch.float32)
            center_x, center_y = width // 2, height // 2
            image[center_y-5:center_y+5, :] = 0.5
            image[:, center_x-5:center_x+5] = 0.5
            marker_size = 20
            image[:marker_size, :marker_size] = 0.8
            image[:marker_size, -marker_size:] = 0.8
            image[-marker_size:, :marker_size] = 0.8
            image[-marker_size:, -marker_size:] = 0.8
            return image.unsqueeze(0)
        elif style == "random":
            return torch.rand((1, height, width, 3), dtype=torch.float32)
        else:
            return torch.zeros((1, height, width, 3), dtype=torch.float32)

    def create_content_latent(self, width, height, style):
        latent_height = height // 8
        latent_width = width // 8
        
        if style == "neutral":
            latent = torch.zeros([1, 4, latent_height, latent_width])
        elif style == "visible":
            latent = torch.ones([1, 4, latent_height, latent_width]) * 0.5
            latent += torch.randn([1, 4, latent_height, latent_width]) * 0.1
        elif style == "random":
            latent = torch.randn([1, 4, latent_height, latent_width])
        else:
            latent = torch.zeros([1, 4, latent_height, latent_width])
        
        return {"samples": latent}

    def create_content_mask(self, width, height, style):
        if style == "neutral":
            return torch.ones((height, width), dtype=torch.float32)
        elif style == "visible":
            mask = torch.ones((height, width), dtype=torch.float32)
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            for y in range(height):
                for x in range(width):
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if dist <= radius:
                        mask[y, x] = 0.0
            return mask
        elif style == "random":
            return torch.rand((height, width), dtype=torch.float32)
        else:
            return torch.ones((height, width), dtype=torch.float32)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "EmptyInputNode": EmptyInputNode,
    "UniversalInputNode": UniversalInputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyInputNode": "📥 空输入节点",
    "UniversalInputNode": "📥 通用输入节点",
}