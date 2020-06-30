# DeepSets

Deep Sets Research Project. UE Recherche Ecole Polytechnique.

This deep learning project provides a scalable framework for Deep Sets architecture in a supervised environment. It provides both Deep Sets Invariant and Deep Sets Equivariant frameworks. Several applications of this architecture are implemented to test its efficiency and performance.

## Architectures.

- Deep Sets Invariant architecture.

- Deep Sets Equivariant architecture.

## Models.

- DeepSets Invariant.
  + DigitSumImage
  + DigitSumText

- DeepSets Equivariant.
  + PointCloud

## Project structure

The project is structured as following:

```bash
.
├── loaders
|  	└── dataset selector
|  	└── digitsum_image_loader.py # loading and pre-processing digitsum_image data
|	└── digitsum_text_loader.py # loading and pre-processing digitsum_text data
|	└── cloudpoints_loader.py # loading and pre-processing cloudpoints data
├── models
|  	└── architecture selector
|  	└── deepsets_invariant.py # generic DeepSets Invariant architecture
|	└── deepsets_invariant_batch.py # generic DeepSets Invariant architecture, better handling batches. Only used in run_batch.py
|	└── deepsets_equivariant.py # generic DeepSets Equivariant architecture
|  	└── digitsum_image.py # digitsum image model
|  	└── digitsum_text.py # digitsum text model
|	└── cloudpoints.py # cloud points model
├── toolbox
|	└── losses.py  # loss selector
|	└── optimizer.py  # optimizer selector
|  	└── logger.py  # keeping track of most results during training and storage to static .html file
|  	└── metrics.py # computing scores and main values to track
|  	└── utils.py   # various utility functions
├── run.py # main file from the project serving for calling all necessary functions for training and testing
├── run_batch.py # slight modification of run.py function, better handling batches
├── args.py # parsing all command line arguments for experiments
├── trainer.py # pipelines for training, validation and testing
├── trainer_batch.py # pipelines for training, validation and testing, slight modification of trainer.py, better handling batches. Only used in run_batch.py
```

## Launching
Experiments can be launched by calling `run.py` and a set of input arguments to customize the experiments. You can find the list of available arguments in `args.py` and some default values. Note that not all parameters are mandatory for launching and most of them will be assigned their default value if the user does not modify them.

Here is a typical launch command and some comments:

- `python3 run.py --name digitsum_image_batch64_val210_reducelronplateau --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 10 --set-weight mean --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.001 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard`
  + this experiment is on the _digitsum_image_ dataset which can be found in `--root-dir/digitsum_image` trained over _digitsum_image50_. It optimizes with _adam_ with initial learning rate (`--lr`) of `1e-3` which is decayed by half whenever the `--scheduler` _ReduceLRonPlateau_ does not see an improvement in the validation accuracy for more than `--step` epochs. In addition it saves intermediate results to `--tensorboard`.
  + if you want to resume a previously paused experiment you can use the `--resume` flag which can continue the training from _best_, _latest_ or a specifically designated epoch.
  + if you want to use your model only for evaluation on the test set, add the `--test` flag.
 
## Output
For each experiment `{name}` a folder with the same name is created in the folder `root-dir/{name}/runs`
 This folder contains the following items:

```bash
.
├── checkpoints (\*.pth.tar) # models and logs are saved every epoch in .tar files. Non-modulo 5 epochs are then deleted.
├── best model (\*.pth.tar) # the currently best model for the experiment is saved separately
├── config.json  # experiment hyperparameters
├── logger.json  # scores and metrics from all training epochs (loss, learning rate, accuracy,etc.)
├── res  # predictions for each sample from the validation set for every epoch
├── tensorboard  # experiment values saved in tensorboard format
 ```

### Tensorboard
In order the visualize metrics and results in tensorboard you need to launch it separately: `tensorboard --logdir /root-dir/{name}/runs`. You can then access tensorboard in our browser at [localhost:6006](localhost:6006)
If you have performed multiple experiments, tensorboard will aggregate them in the same dashboard.
  
  
 ## Requirements
 - Python 3
 - Pytorch
 - Tensorboard 1.14