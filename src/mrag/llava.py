import copy
import re

import torch


def load_llava(args, load_pretrained_model, log):
    model_name = "llava_qwen"
    visible_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dm_arg = str(args.llava_device_map).strip().lower()
    if dm_arg == "auto" and visible_cuda_count == 1:
        llava_device_map = {"": "cuda:0"}
        log("llava_device_map auto->single (single GPU detected)")
    elif dm_arg in ("single", "cuda", "cuda:0", "0"):
        llava_device_map = {"": "cuda:0"}
    elif re.fullmatch(r"(?:cuda:)?\d+", dm_arg):
        device_idx = dm_arg.split(":")[-1]
        llava_device_map = {"": f"cuda:{device_idx}"}
    else:
        llava_device_map = args.llava_device_map
    if llava_device_map == "auto" and visible_cuda_count >= 2:
        llava_device_map = "balanced"
        log("llava_device_map auto->balanced (multi-GPU detected)")

    llava_args = {
        "multimodal": True,
        "overwrite_config": {"image_aspect_ratio": "pad"},
    }
    if args.llava_attn_implementation:
        llava_args["attn_implementation"] = args.llava_attn_implementation
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.llava_model_path,
        None,
        model_name,
        device_map=llava_device_map,
        load_4bit=bool(args.llava_load_4bit),
        load_8bit=bool(args.llava_load_8bit),
        **llava_args,
    )
    model.eval()
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        counts = {}
        for v in device_map.values():
            key = str(v)
            counts[key] = counts.get(key, 0) + 1
        summary = ", ".join([f"{k}:{counts[k]}" for k in sorted(counts.keys())])
        log(f"llava_hf_device_map={summary}")
        has_cpu_offload = any(str(v) == "cpu" for v in device_map.values())
        if has_cpu_offload and not args.llava_allow_cpu_offload:
            raise RuntimeError(
                "LLaVA CPU offload detected and llava_allow_cpu_offload is disabled. "
                "Try one of: CUDA_VISIBLE_DEVICES=0,1 or LLAVA_DEVICE_MAP=single or LLAVA_LOAD_4BIT=1."
            )
    return tokenizer, model, image_processor


def llava_answer(
    tokenizer,
    model,
    image_processor,
    item,
    image_files,
    args,
    default_image_token,
    image_token_index,
    conv_templates,
    tokenizer_image_token,
    process_images,
):
    question_part = item.get("prompt_question_part", item["question"])
    if len(image_files) <= 1:
        instruction = "Answer with the option's letter from the given choices directly. "
    else:
        instruction = (
            "You will be given one question concerning several images. "
            "The first image is the input image, others are retrieved examples to help you. "
            "Answer with the option's letter from the given choices directly. "
        )
    image_tokens = default_image_token * max(1, len(image_files))
    user_query = f"{instruction}{image_tokens}\n{question_part}"

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], user_query)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    model_device = next(model.parameters()).device
    image_dtype = torch.float16 if model_device.type == "cuda" else torch.float32
    input_ids = (
        tokenizer_image_token(prompt_question, tokenizer, image_token_index, return_tensors="pt")
        .unsqueeze(0)
        .to(model_device)
    )
    image_tensors = process_images(image_files, image_processor, model.config)
    image_tensors = [img.to(dtype=image_dtype, device=model_device) for img in image_tensors]
    image_sizes = [img.size for img in image_files]

    with torch.inference_mode():
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.0,
            num_beams=max(1, int(args.llava_num_beams)),
            max_new_tokens=max(1, int(args.llava_max_new_tokens)),
        )
    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
