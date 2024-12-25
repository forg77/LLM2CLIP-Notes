## LLM2CLIP-Notes
The official LLM2CLIP (https://github.com/microsoft/LLM2CLIP) provides contrastive learning training code based on llm2vec and includes modified CLIP code based on EVA-CLIP (https://github.com/baaivision/EVA/tree/master/EVA-CLIP). However, there are still some parts of the code missing. Below are some modifications and additions needed during the reproduction process.

## Points to NOte
1. The environment for EVA-CLIP is based on CUDA 11.6, and some versions of previous packages, such as xformers, need to be adjusted. See eva_clip_pip_list.txt for details.

2. Data can be processed into .tar package format according to the webdataset format. Pay attention to the order of the training data when packaging.

3. In the get_wds_dataset function in training/data.py, I made some modifications to get_text, but the dataloader could not iterate normally as it found no data. Therefore, I commented out the related code of get_text and made some modifications in training/train.py.

```python
#training/data.py
    def get_text(item):
        sample = {}
        # print(item['__key__'])
        sample['image'] = item['jpg']
        if item.get('short_caption',-1) != -1:
            f1 = np.frombuffer(item['short_caption'], dtype=np.float32)
        else:
            f1 = None
        if item.get('long_caption',-1) != -1:
            f2 = np.frombuffer(item['long_caption'], dtype=np.float32)
        else:
            f2 = None
        if f1 is not None and f2 is not None:
            sample['text'] = random.choice([f1, f2])
        elif f1 is not None:
            sample['text'] = f1
        elif f2 is not None:
            sample['text'] = f2

    pipeline.extend([
        # wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        # wds.rename(image="jpg;png;jpeg;webp",),
        # wds.map(get_text),
        wds.map_dict(jpg=preprocess_img),
        wds.to_tuple('jpg', 'short_caption','long_caption'), 
        wds.batched(args.batch_size, partial=not is_train),
    ])
# My dataset includes three fields: short_caption, long_caption, jpg
```

```python
#training/train.py
for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            scheduler(step)

        images, short_c, long_c = batch
        texts = random.choice([short_c, long_c])

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = np.array([np.frombuffer(item, dtype=np.float32) for item in texts]) # Convert bytes to float32 numpy array
        texts = torch.from_numpy(texts)
        texts = texts.to(device=device, dtype=cast_dtype, non_blocking=True)

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()
```




## Acknowledgement

Many thanks for the work on LLM2CLIP and EVA-CLIP.




## Citation

```bibtex
@misc{huang2024llm2clippowerfullanguagemodel,
      title={LLM2CLIP: Powerful Language Model Unlock Richer Visual Representation}, 
      author={Weiquan Huang and Aoqi Wu and Yifan Yang and Xufang Luo and Yuqing Yang and Liang Hu and Qi Dai and Xiyang Dai and Dongdong Chen and Chong Luo and Lili Qiu},
      year={2024},
      eprint={2411.04997},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.04997}, 
}
```