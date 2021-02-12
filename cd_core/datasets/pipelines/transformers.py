import random

from torchvision import transforms
from torchvision.transforms import functional as tvF


from ..builder import PIPELINES

# TODO(snian) add more transformers
              
@PIPELINES.register_module()
class ToTensor(transforms.ToTensor):
  def __call__(self,pics):
    return [tvF.to_tensor(pic) for pic in pics]

@PIPELINES.register_module()
class ToPILImage(transforms.ToPILImage):
  def __call__(self,imgs):
    return [tvF.to_pil_image(img,self.mode) for img in imgs]

@PIPELINES.register_module()
class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
  def __call__(self,imgs):
    if random.random() < self.p:
      return [tvF.hflip(img) for img in imgs]
    return imgs

@PIPELINES.register_module()
class RandomVerticalFlip(transforms.RandomVerticalFlip):
  def __call__(self,imgs):
    if random.random() < self.p:
      return [tvF.vflip(img) for img in imgs]
    return imgs

