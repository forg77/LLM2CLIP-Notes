## LLM2CLIP-Notes
LLM2CLIP官方(https://github.com/microsoft/LLM2CLIP)给出了基于llm2vec的对比学习训练代码，并且给出了基于EVA-CLIP(https://github.com/baaivision/EVA/tree/master/EVA-CLIP)修改的clip代码。但是仍然有小部分代码缺失，以下是在复现时发现的一些需要修改和补充的地方。

## 需要注意的点
1. evaclip的环境时基于cuda11.6的，需要调整一些过去的包的版本，如xformers等，详见`eva_clip_pip_list.txt`。

2. 可以按照webdataset的格式把数据处理成.tar包的形式，注意打包时的训练数据的顺序。

3. training/data.py中`get_wds_dataset`函数中，我对`get_text`做了一些修改，但是dataloader不能正常迭代，发现其中没有数据。所以注释掉了`get_text`相关的代码，在training/train.py中做了一些修改。


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
#我的数据集包括了 short_caption, long_caption, jpg三个字段
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
        texts = np.array([np.frombuffer(item, dtype=np.float32) for item in texts]) # 从bytes转为float32 numpy array
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

感谢LLM2CLIP和EVA-CLIP的工作




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