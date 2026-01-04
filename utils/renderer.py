import torch
from torch import nn
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
)


def hard_channel_blend(
    colors: torch.Tensor, fragments,
) -> torch.Tensor:
    """
    简单地将前K个面的颜色混合，返回C+1通道图像
    Args:
        colors: (N, H, W, K, C) 每个像素前K个面的RGB颜色
        fragments: 光栅化的输出。从中我们使用
            - pix_to_face: 形状为(N, H, W, K)的LongTensor，指定
              与图像中每个像素重叠的面索引（在打包表示中）。这用于
              确定输出形状
    Returns:
        RGBA像素通道: (N, H, W, C+1)
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device

    # 背景遮罩
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)

    background_color = torch.ones(colors.shape[-1], dtype=colors.dtype, device=colors.device)

    # 找出需要扩展多少背景颜色以用于masked_scatter
    num_background_pixels = is_background.sum()

    # 设置背景颜色
    pixel_colors = colors[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, C)

    # 与alpha通道连接
    alpha = (~is_background).type_as(pixel_colors)[..., None]

    return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, C+1)


class SimpleShader(nn.Module):
    """
    简单着色器类
    """
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        images = hard_channel_blend(texels, fragments)
        return images


class MeshRendererWithDepth(nn.Module):
    """
    带深度的网格渲染器
    """
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


class Renderer(nn.Module):
    """
    渲染器类
    """
    def __init__(self):
        super().__init__()
        self.raster_settings = None

    def set_rasterization(self, cameras):
        """
        设置光栅化参数
        :param cameras: 相机对象
        """
        image_size = tuple(cameras.image_size[0].detach().cpu().numpy())
        self.raster_settings = RasterizationSettings(
            image_size=(int(image_size[0]), int(image_size[1])),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

    def forward(self, input):
        mesh = input["mesh"]
        cameras = input["cameras"]
        if self.raster_settings is None:
            self.set_rasterization(cameras)

        mesh_renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SimpleShader()
        )
        images, depth = mesh_renderer(mesh)
        return images, depth


class RendererBev(nn.Module):
    """
    BEV（鸟瞰图）渲染器类
    """
    def __init__(self):
        super().__init__()
        self.raster_settings = None
        self.image_size = tuple((640, 1024))  # FOV相机没有image_size

    def set_rasterization(self):
        """
        设置光栅化参数
        """
        image_size = self.image_size
        self.raster_settings = RasterizationSettings(
            image_size=(int(image_size[0]), int(image_size[1])),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

    def forward(self, input):
        mesh = input["mesh"]
        cameras = input["cameras"]
        if self.raster_settings is None:
            self.set_rasterization()

        mesh_renderer = MeshRendererWithDepth(
            rasterizer=MeshRenderer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SimpleShader()
        )
        images, depth = mesh_renderer(mesh)
        return images, depth
