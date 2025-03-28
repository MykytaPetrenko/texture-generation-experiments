import nodes
import comfy.model_base
import torch
import comfy.sample
import comfy.samplers
import time

class BatchPromptTemplate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_template": ("STRING", {"multiline": True}),
                "batch_size": ("INT", {"default": 2})
            }
        }
    
    RETURN_TYPES = ("BATCH_PROMPTS",)
    RETURN_NAMES = ("batch_prompts",)
    FUNCTION = "process_prompts"
    CATEGORY = "prompt_tools/conditioning"

    def process_prompts(self, prompt_template, batch_size):
        outputs = [prompt_template] * batch_size
        return (outputs, )
    

class BatchPromptInsertByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_prompts": ("BATCH_PROMPTS",),
                "insertion_key": ("STRING", {"default": "{placeholder}"}),
                "indices": ("STRING", {"default": "1,2"}),
                "insertion": ("STRING", {"multiline": True})
            }
        }
    
    RETURN_TYPES = ("BATCH_PROMPTS",)
    RETURN_NAMES = ("batch_prompts",)
    FUNCTION = "process_prompts"
    CATEGORY = "prompt_tools/conditioning"

    def process_prompts(self, batch_prompts, insertion_key, indices, insertion):
        if indices == "":
            # Insert to all prompts if indices are not specified
            indices_to_process = [i for i in range(len(batch_prompts))]
        else:
            indices_arr = indices.split(",")
            indices_to_process = set()
            for idx_str in indices_arr:
                idx = int(idx_str)
                if idx >= len(batch_prompts) or idx < 0:
                    raise Exception(f"Index {idx} out of range ({len(batch_prompts)})")
                
                indices_to_process.add(idx)
        
        outputs = batch_prompts.copy()
        for idx in indices_to_process:
            temp = outputs[idx]
            if insertion_key in temp:
                temp = temp.replace(insertion_key, insertion)
                outputs[idx] = temp
            else:
                print(
                    f"WARNING: insertion key \"{insertion_key}\" in not found in template string \"{temp}\""
                )

        return (outputs, )


class ClipTextEncodeBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_prompts": ("BATCH_PROMPTS",),
                "clip_model": ("CLIP",)
            }
        }

    RETURN_TYPES = ("BATCH_CONDITIONING",)
    RETURN_NAMES = ("batch_conditionings",)
    FUNCTION = "process_prompts"
    CATEGORY = "prompt_tools/conditioning"

    def process_prompts(self, batch_prompts, clip_model):        
        # CLIP Text Encode each prompt
        batch_conditions = []
        for prompt in batch_prompts:
            if prompt:
                # Use the provided CLIP model to encode the prompt
                # This mimics the behavior of a standard CLIP Text Encode node
                text_encode_node = nodes.CLIPTextEncode()
                conditioning = text_encode_node.encode(clip_model, prompt)[0]
                batch_conditions.append(conditioning)
            else:
                # If no prompt, return a default/empty conditioning
                batch_conditions.append(None)
        
        return (batch_conditions,)

class BatchControlNetApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_batch": ("BATCH_CONDITIONING",),
                "negative_batch": ("BATCH_CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                "vae": ("VAE", ),
            }

        }

    RETURN_TYPES = ("BATCH_CONDITIONING", "BATCH_CONDITIONING")
    RETURN_NAMES = ("positive_batch", "negative_batch")
    FUNCTION = "apply_controlnet"
    CATEGORY = "conditioning/batch"

    def apply_controlnet(self, positive_batch, negative_batch, control_net, image, strength, start_percent, end_percent, vae=None):
        if len(positive_batch) != len(negative_batch):
            raise ValueError(f"Number of positive conditions ({len(positive_batch)}) must match number of negative conditions ({len(negative_batch)})")
        
        # Get number of images in the batch
        image_batch_size = image.shape[0]
        cond_batch_size = len(positive_batch)
        
        print(f"Image batch size: {image_batch_size}, Conditioning batch size: {cond_batch_size}")
        
        # Use the smaller of the two batch sizes
        max_batch = min(image_batch_size, cond_batch_size)
        print(f"Will process {max_batch} items")
        
        controlnet_node = nodes.ControlNetApplyAdvanced()
        processed_positives = []
        processed_negatives = []
        
        # Apply control net to each pair of conditions with corresponding image
        for i in range(max_batch):
            pos = positive_batch[i]
            neg = negative_batch[i]
            
            # Get the corresponding image for this batch item
            current_image = image[i:i+1]
            
            # Apply controlnet to current image and conditioning pair
            pos_result, neg_result = controlnet_node.apply_controlnet(pos, neg, control_net, current_image, strength, start_percent, end_percent, vae)
            processed_positives.append(pos_result)
            processed_negatives.append(neg_result)
            print(f"Processed ControlNet for batch item {i+1}/{max_batch}")
        
        return (processed_positives, processed_negatives)

class KSamplerBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive_batch": ("BATCH_CONDITIONING",),
            "negative_batch": ("BATCH_CONDITIONING",),
            "latent_image": ("LATENT",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/batch"

    def sample(self, model, positive_batch, negative_batch, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise):
        import comfy.sample
        import comfy.samplers
        
        if len(positive_batch) != len(negative_batch):
            raise ValueError(f"Number of positive conditions ({len(positive_batch)}) must match number of negative conditions ({len(negative_batch)})")

        # Get original latent samples
        original_samples = latent_image["samples"]
        batch_size = len(positive_batch)
        
        # Check if input latent already has multiple images
        input_batch_size = original_samples.shape[0]
        print(f"Input latent has shape {original_samples.shape} ({input_batch_size} images)")
        print(f"Processing {batch_size} conditioning pairs")
        
        # Determine max batch size - use min of conditions and input images
        max_batch = min(batch_size, input_batch_size)
        print(f"Will process {max_batch} items (min of conditioning pairs and input images)")
        
        # Create output dictionary
        output_latent = latent_image.copy()
        
        # Initialize with empty result tensor matching original latent channels/dimensions
        c, h, w = original_samples.shape[1:]
        result_shape = (max_batch, c, h, w)
        
        # Create result tensor
        result_samples = torch.zeros(result_shape, dtype=original_samples.dtype, device=original_samples.device)
        
        # Process each conditioning pair with its corresponding latent
        for i in range(max_batch):
            positive = positive_batch[i]
            negative = negative_batch[i]
            
            if positive is None or negative is None:
                print(f"Warning: Skipping batch item {i} due to None conditioning")
                continue
                
            try:
                # Get the corresponding latent image for this batch item
                current_latent_image = original_samples[i:i+1].clone()  # Keep as [1, c, h, w]
                current_seed = seed + i
                
                # Fix empty latent channels
                current_latent_image = comfy.sample.fix_empty_latent_channels(model, current_latent_image)
                
                # Prepare noise with incrementing seed
                batch_inds = latent_image.get("batch_index", None)
                if batch_inds is not None:
                    # Take only relevant batch indices if available
                    batch_inds = batch_inds[i:i+1] if i < len(batch_inds) else None
                    
                noise = comfy.sample.prepare_noise(current_latent_image, current_seed, batch_inds)
                
                # Get noise mask for this batch if available
                noise_mask = None
                if "noise_mask" in latent_image:
                    if latent_image["noise_mask"].shape[0] > i:
                        noise_mask = latent_image["noise_mask"][i:i+1]
                
                # Show which batch item we're processing in the console
                print(f"Sampling batch item {i+1}/{max_batch}")
                start_time = time.time()
                
                # Simple console progress callback
                def callback(step, x0, x, total_steps):
                    # Print a simple progress indicator to the console
                    percent = int((step + 1) / total_steps * 100)
                    print(f"Item {i+1}/{max_batch}: Step {step + 1}/{total_steps} ({percent}%)", end="\r")
                    return x
                
                # Perform sampling
                samples = comfy.sample.sample(
                    model, 
                    noise, 
                    steps, 
                    cfg, 
                    sampler_name, 
                    scheduler, 
                    positive, 
                    negative, 
                    current_latent_image,
                    denoise=denoise, 
                    noise_mask=noise_mask, 
                    callback=callback,
                    disable_pbar=True, 
                    seed=current_seed
                )
                
                # Store result in our batch
                result_samples[i] = samples[0]
                elapsed = time.time() - start_time
                print(f"\nCompleted batch item {i+1}/{max_batch} in {elapsed:.2f}s")
                
            except Exception as e:
                print(f"Error processing batch item {i+1}/{max_batch}: {e}")
                import traceback
                traceback.print_exc()

        # Update the output latent with our batched results
        output_latent["samples"] = result_samples
        
        # Handle other batch-related fields like noise_mask
        if "noise_mask" in output_latent and output_latent["noise_mask"].shape[0] != max_batch:
            if output_latent["noise_mask"].shape[0] >= max_batch:
                output_latent["noise_mask"] = output_latent["noise_mask"][:max_batch]
            else:
                # Pad if needed (should be rare)
                pad_shape = list(output_latent["noise_mask"].shape)
                pad_shape[0] = max_batch - output_latent["noise_mask"].shape[0]
                padding = torch.zeros(pad_shape, device=output_latent["noise_mask"].device, 
                                     dtype=output_latent["noise_mask"].dtype)
                output_latent["noise_mask"] = torch.cat([output_latent["noise_mask"], padding], dim=0)
                
        # Same for batch_index if it exists
        if "batch_index" in output_latent and len(output_latent["batch_index"]) != max_batch:
            if len(output_latent["batch_index"]) >= max_batch:
                output_latent["batch_index"] = output_latent["batch_index"][:max_batch]
            # For batch_index we just let it be if it's shorter
        
        print(f"Final batched latent shape: {result_samples.shape}")
        return (output_latent,)

NODE_CLASS_MAPPINGS = {
    "BatchPromptTemplate": BatchPromptTemplate,
    "BatchPromptInsertByIndex": BatchPromptInsertByIndex,
    "ClipTextEncodeBatch": ClipTextEncodeBatch,
    "BatchControlNetApply": BatchControlNetApply,
    "KSamplerBatch": KSamplerBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchPromptTemplate": "Batch Prompt Template",
    "BatchPromptInsertByIndex": "Batch Prompt Insert By Index",
    "ClipTextEncodeBatch": "Clip Text Encode Batch",
    "BatchControlNetApply": "Batch ControlNet Apply",
    "KSamplerBatch": "KSampler Batch"
}
