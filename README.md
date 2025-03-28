# texture-generation-experiments

## Work in Progress. Texture generation with Hunyuan3d Paint and SDXL wildcards.

## Models 
Model used for refining: [WildCardX-XL TURBO](https://civitai.com/models/293331/wildcardx-xl-turbo) with [VAE](https://civitai.com/models/296576/sdxl-vae)
Depth and Canny Control LoRAs can be found [here](https://huggingface.co/stabilityai/control-lora/tree/main)
Use any upscaler you like. For example Valar, Ultrasharp, ClearReality

## How to use
1. Install all necessary custom nodes via Manager
2. Install my node. **Just put py file from the repository to you custom nodes folder** (I will improve installation process later).
3. Make sure to download all models except those which will be downloaded automatically (has prefix (Down))
4. Upload reference image and glb model file (I render depth in blender to generate references).
5. Write **positive** and **negative** prompts. I use my custom nodes with allows to generate different images of the batch with different prompts. Thats why **keep placeholder {x}** in your prompt, so additional information regarding camera angle will be inserter. Otherwise it may draw a face on the back of the head etc.
6. "!!!View size" input refers to the multiview resolution. Multiview generation requires a lot of VRAM. with 1024 it eats ~9-10 GB with 4 projections only. But even 512 often is enough as the view projections will be refined and upscaled with SDXL model further.
7. If you don't have a lot of VRAM you can get OOM on the refining step. I feed 2048 x 2048 for refining. You may tweak the resolution changing width and height in the "Upscale Image" green node (Refine Multiview group)
8. Queue prompt and wait for result. It is not too fast.
9. Also **if you don't have sharp skin details** like scales you may lower canny influence or completelly remove it from the workflow.


## Tested on 16 gigs of VRAM. But used VRAM goes up only to around 12 Gb. If you have low VRAM, lower down resolutions. See 7 and 6. Also you may use tiled VAE encode/decode. They works much slower, but lower down VRAM usage

