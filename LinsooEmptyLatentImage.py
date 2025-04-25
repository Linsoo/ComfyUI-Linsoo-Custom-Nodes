import re
import torch
import comfy.model_management

#   " "", "", "","1600 x 1280", "1856 x 1088"
class LinsooEmptyLatentImage:
    RECOMMEND_RESOLUTION=['640x1536 (5:12)',
                          '768x1344 (4:7)',
                          '832x1216 (13:19)',
                          '896x1152 (7:9)',
                          '1088x1856 (17:29)',
                          '1248x1824 (13:19)',
                          '1280x1600 (4:5)',
                          '1344x1536 (7:8)',

                          '512x512 (1:1)',
                          '1024x1024 (1:1)',
                          '1472x1472 (1:1)',
                          '1536x1536 (1:1)',

                          '1152x896 (9:7)',
                          '1216x832 (19:13)',
                          '1248x1824 (13:19)',
                          '1344x768 (7:4)',
                          '1536x640 (12:5)',
                          '1536x1344 (8:7)',
                          '1600x1280 (5:4)',
                          '1856x1088 (29:17)',
                          ]
                          
                          
                          
                          

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "recommend_resolution": (s.RECOMMEND_RESOLUTION,{'default': s.RECOMMEND_RESOLUTION[4],'tooltip': "Select the recommended resolution."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "generate"

    CATEGORY = "linsoo"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def generate(self, recommend_resolution:str = None, batch_size=1):

        regex = r"(\d{2,})x(\d{2,})"
        matches = re.search(regex, recommend_resolution, re.IGNORECASE)
        if matches:
            gr = matches.groups()
            latent = torch.zeros([batch_size, 4, int(gr[1]) // 8, int(gr[0]) // 8], device=self.device)
        else:
            latent = torch.zeros([batch_size, 4, 512 // 8, 512 // 8], device=self.device)
        return ({"samples":latent}, )
