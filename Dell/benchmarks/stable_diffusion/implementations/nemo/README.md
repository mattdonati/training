## Running NVIDIA NeMo Stable Diffusion PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA NeMo Stable Diffusion PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 4TB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- Dataset preprocessing requires access to a single node with 8 GPUs.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:stable_diffusion-pyt
```

### 3.2 Download and preprocess datasets and checkpoints

Start the container, replacing `</path/to/your/datasets>` and `</path/to/your/checkpoints>` with existing paths to where you want to save datasets and checkpoints:

```bash
docker run -it --rm --gpus all --network=host --ipc=host --volume </path/to/your/datasets>:/datasets --volume </path/to/your/checkpoints>:/checkpoints <docker/registry>/mlperf-nvidia:stable_diffusion-pyt
# now you should be inside the container in the /workspace/sd directory
```

The benchmark employs two datasets:

1. Training: a subset of [laion-400m](https://laion.ai/blog/laion-400-open-dataset)
2. Validation: a subset of [coco-2014 validation](https://cocodataset.org/#download)

#### 3.2.1 Download Laion 400m

The benchmark uses a CC-BY licensed subset of the Laion400 dataset.

The LAION datasets comprise lists of URLs for original images, paired with the ALT text linked to those images. As
downloading millions of images from the internet is not a deterministic process and to ensure the replicability of the
benchmark results, hence we download it from MLCommons storage. The dataset can be downloaded using a script:

```bash
bash scripts/datasets/laion400m-filtered-download-moments.sh
```

Total download size: ~831GB. Assuming download speed 1 GB/min, it takes ~14 hours to complete.

#### 3.2.2 Download and preprocess COCO-2014

The COCO-2014-validation dataset consists of 40,504 images and 202,654 annotations. However, our benchmark uses only a
subset of 30,000 images and annotations chosen at random with a preset seed. It's not necessary to download the entire
COCO dataset as our focus is primarily on the labels (prompts) and the inception activation for the corresponding
images (used for the FID score).

We download this dataset from MLCommons storage using the following scripts:

```bash
bash scripts/datasets/coco2014-validation-download-prompts.sh --output-dir /datasets/coco2014  # downloads instantly
bash scripts/datasets/coco2014-validation-download-stats.sh --output-dir /datasets/coco2014  # downloads instantly
bash scripts/datasets/coco-2014-validation-download.sh  # downloads ~6GB to /datasets/coco2014 and unzips
bash scripts/datasets/coco-2014-validation-split-resize.sh  # writes to /datasets/coco2014/val2014_512x512_30k
```

Total size after preprocessing: ~15GB.

#### 3.2.3 Download checkpoints

The benchmark utilizes several network architectures for both the training and validation processes:

1. **Stable Diffusion**: This component leverages StabilityAI's 512-base-ema.ckpt checkpoint from HuggingFace. While the
   checkpoint includes weights for the UNet, VAE, and OpenCLIP text embedder, the UNet weights are not used and are
   discarded when loading the weights. The checkpoint can be downloaded with the following command:

```bash
bash scripts/checkpoints/download_sd.sh --output-dir /checkpoints/sd  # downloads ~4.9GB
```

2. **Inception**: The Inception network is employed during validation to compute the Fréchet Inception Distance (FID)
   score. The necessary weights can be downloaded with the following command:

```bash
bash scripts/checkpoints/download_inception.sh --output-dir /checkpoints/inception  # downloads <100MB
```

3. **OpenCLIP ViT-H-14 Model**: This model is utilized for the computation of the CLIP score. The required weights can
   be downloaded using the command:

Note: the script `scripts/checkpoints/download_clip.sh` is deprecated. Instead, use the method below.

```bash
python -c "from infer_and_eval_tools import CLIPEncoder; clip_model = CLIPEncoder(clip_version='ViT-H-14', cache_dir='/checkpoints/clip', device='cpu')"  # downloads ~3.7GB
```

The aforementioned scripts will handle both the download and integrity verification of the checkpoints.

#### 3.2.4 Preprocess Laion 400m dataset

```bash
bash scripts/datasets/laion400m-encode-captions.sh
```

This script creates a version of dataset that contains encoded CLIP captions.
It requires single node with preferably 8 GPUs and 2.7TB space for outputs.
There are 832 shards to process. Splitting the work across 8 GPUs and assuming each fragment takes ~100 seconds to process, the total time needed is estimated at ~3 hours.


### 3.3 Final datasets and checkpoints structure

(showing only relevant files)

```
/datasets/laion-400m/webdataset-moments-filtered-encoded  # 2.7T containing 832 tar files inside
/datasets/coco2014/val2014_30k.tsv  # 2M
/datasets/coco2014/val2014_512x512_30k_stats.npz  # 31M
/checkpoints/clip/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/blobs/9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4  # 3.7G
/checkpoints/inception/pt_inception-2015-12-05-6726825d.pth  # 92M
/checkpoints/sd/512-base-ema.ckpt  # 4.9G
```

Exit the container.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:

```bash
export DATADIR="/path/to/your/laion/dataset"
export COCODIR="/path/to/your/coco/dataset"
export CHECKPOINT_CLIP="/path/to/your/clip/checkpoint"
export CHECKPOINT_FID="/path/to/your/fid/checkpoint"
export CHECKPOINT_SD="/path/to/your/sd/checkpoint"
export LOGDIR="/path/to/output_logdir"  # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:stable_diffusion-pyt
source config_<system>.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```

- Sourcing config loads hyperparameters as environment variables.
- Replace "/path/to/your/" with the real path.

## 5. Evaluation

### Quality metric

The quality metric in this benchmark is FID score and CLIP score.

### Quality target

* The target FID is below or equal to 90.
* The target CLIP is above or equal to 0.15.

### Evaluation frequency

The evaluation schedule is the following:

- Evaluate every 512000 training samples

Evaluation time does not count into the total runtime. It can be performed after the training is finished.
