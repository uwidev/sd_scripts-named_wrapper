## Abstract
This is a [sd-scripts](https://github.com/kohya-ss/sd-scripts) (in conjunction with [Lora Easy Training Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts?tab=readme-ov-file)) wrapper for organizing the outputs of your various trainings.

It will gather information from your configuration files to create a reference name that can be read at a glance to determine what training parameters were used.

For example:
```
umbrella-mnoobv-oCAME-u4e6-b4-e32-alocon-d32a16cd16ca8-db-v4.safetensor
```
This reads from left to right as follows:
- concept is umbrella
- training on model Noobs NAI V-Prediction, shortened to noobv
- with CAME optimizer
- at UNET lr 4e-5 (notice the absence of `t#e#`, implying TE is not trained)
- with batch 4
- for 32 epochs
- using network algorithm locon
- with dim 32, alpha 16, convolution dim 16, convolution alpha 8
- with debias estimation loss
- using the v4 dataset

There are more elements to add to the name, like training resolution, but by default it assumes sdxl 1024, so it doesn't include that information. In other words, training parameters that are *default* are not included in the name.

This name is used to do a python str.format on all settings. The initial string taken from `output_name` in your configuration is the `{basename}`. It should be what you are training. The compiled name corroponds to `{name}`. Use these placeholders to dynamically set the output path and logging name.

For example...
```toml
[Save]
output_dir = /home/username/lora-training/{basename}/out/{name}
```
...gets turned into...
```toml
[Save]
output_dir = /home/username/lora-training/umbrella/out/umbrella-mnoobv-oCAME-u4e6-b4-e32-alocon-d32a16cd16ca8-db-v4
```

The configuration's `output_name` setting will be overwritten by the compiled name at the very end. It should NOT have any placeholders.

## Usage
```sh
python train_lora.py <jobs_path>
```
...where jobs_path is a directory of jobs to iterate over and to train.

When jobs are finished training, the configurations will be moved to `../archive/{basename}`.

### Setup
The script was designed to be used somewhat in conjunction with [Lora Easy Training Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts?tab=readme-ov-file), but is more a more primitive, automated, cli approach. The script should be placed at the root directory. Otherwise make modifications as needed.

The script starts training formatted as follows...
```sh
./backend/sd_scripts/sdxl_train_network.py --dataset_config ./backend/runtime_store/dataset.toml --config_file ./backend/runtime_store/config.toml
```

You need a configuration file that defines the dataset and another one for training parameters. You should use [KohakuBlueleaf's example configurations](https://github.com/KohakuBlueleaf/LyCORIS/tree/main/example_configs/training_configs/kohya) as a template and adjust it to your training needs. Do not re-organize items away from their headings/groupings.

Your folder setup should be as follows...
```
 /home/username/jobs/queue
└──  concept
    ├──  config.toml
    └──  dataset.toml
```

And you would call this path as follows.
```sh
python train_lora.py /home/username/jobs/queue
```
