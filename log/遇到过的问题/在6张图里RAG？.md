2/6

常规样本通常是 1 张 query + 5 张候选
Incomplete 场景会被裁成 1 张 query + 1 张候选

```python
        ### our evaluation instuction for all the models 
        if not args.use_rag: 
            prompt = f"Answer with the option's letter from the given choices directly. {image_placeholder}\n"
            image_files = [image]
        else: 
            image_files = [image] + gt_images
            prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly. {image_placeholder}{image_placeholder}{image_placeholder}{image_placeholder}{image_placeholder}{image_placeholder}\n"
            if scenario == 'Incomplete':
                prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly. {image_placeholder}{image_placeholder}\n"

```

去掉第 1 张 query 图之后剩下的候选图”
rag_images = image_files[1:]



Incomplete 场景有特殊规则，不管是 gt_images 还是 retrieved_images，都会先裁成 1 张