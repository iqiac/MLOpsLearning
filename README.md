# MLOpsLeaning

This repository is a small collection consisting of machine learning practices.
It demonstrates basic knowledge in machine learning and common tools used for machine learning projects.

## Development Environment

For consistency, the project runs in a containerized environment.
Requirements:
- [Docker, Docker-Compose](https://docs.docker.com/desktop/setup/install/linux/ubuntu/)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if GPU-acceleration is available and desired
- [VS Code](https://code.visualstudio.com/download)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

> Of course there are other ways to create a consistent development environment. This is only a suggestion.

### Docker

The dependencies available in the environment are described inside `Dockerfile`.
The base image uses micromamba package manager for conda-compatibility, speed, and minimal default packages.
The base image also contains cuda runtime to enable ease the installation of CUDA-enabled Pytorch.

> Depending on the Nvidia driver, the base image tag and Pytorch installation might require some changes.

The host user is added and used inside the container, so the rights to mounted volumes are consistent inside the container and on the host machine.

There are two Docker stages:
- `build`: intended to be minimal and to have all the dependencies needed for CI/CD
- `dev`: intended to have all the tools that can help the development process

`docker-compose.yml` configures the build and deployment process.

### Dev Container

On top of the Docker setup, `.devcontainer/devcontainer.json` configures VS-Code environment.
Additional extensions can be added here.

### VS Code Configuration

`settings.json` configures settings for VS-Code and extensions.

`launch.json` contains debug configurations.

### Python Configuration

Installed tools related to Python (not VS-Code extensions) can be configured in `pyproject.toml`.

## Notebooks

These are simple Jupyter notebooks that can be run cell by cell.

`classificaton.ipynb` showcases different classifiers in a binary classification setting,
where the dataset is rather small with a small class imbalance rate.
It further shows the application of techniques like cross validation and hyperparameter optimization using [Scikit-Learn](https://scikit-learn.org/stable/index.html)
and visualization using [Matplotlib](https://matplotlib.org/).

`regression.ipynb` starts with data analysis using [Pandas](https://pandas.pydata.org/).
It continues to showcase how to handle incomplete data using Scikit-Learn functionalities,
and then applying cross validation and hyperparameter optimization to optimize the results of the trained model which is embedded in a pipeline.

## Unet

This showcases how a deep learning project can be modularly structured without Jupyter notebooks,
making it command-line friendly, when one has to work on a remote server.

- `Unet/config/config.json`: Central configuration file for the whole pipeline.
- `Unet/src`: Contains all the Python modules
    - `dataset.py`: Implements dataset used during training and inference.
    - `model.py`: Implements Unet neural network for segmentation tasks.
    - `trainer.py`: Implements the whole training process.
    - `predictor.py`: Implements the inference process.
    - `main.py`: Entrypoint, starts the process depending on configuration.

There are also `Unet/data` and `Unet/weights` that are ignored by Git
but tracked via [Data Version Control (DVC)](https://dvc.org/).

To run the process:
1. Get data using `dvc pull`
2. Navigate to `Unet` via `cd Unet`
3. Run `python main.py`
    - With argument `--config <path/to/some/config/file>` for specific config file
    - With argument
        - `--mode train` if only training desired
        - `--mode predict` if only inference desired
        - `--mode all` if both desired
    - The training mode can also be configured in config file

Training mode will train the model, save its weights (if save path given),
and plot the training and validation performance.

Inference mode will take the given weights and predict on given directory with images.
This will produce image triplets of orignal image, predicted segmentation mask, and an overlay.
