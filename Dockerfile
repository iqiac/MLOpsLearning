# Image uses cuda 12.6, depending on the machine, you might need to change
# - the cuda version
# - the pytorch version
FROM mambaorg/micromamba:2-cuda12.6.3-ubuntu24.04 AS build

# Install python packages
RUN micromamba install -n base -y \
  -c conda-forge \
  python \
  poetry \
  pytest \
  pylint \
  black \
  isort \
  notebook \
  pandas \
  matplotlib \
  scikit-learn \
  seaborn \
  opencv \
  dvc
RUN micromamba update --all -y && micromamba clean --all -y

# pytorch-cuda 12.6 needs special care,
# cf. https://pytorch.org/get-started/locally/
RUN micromamba run -n base \
  pip install \
    --index-url https://download.pytorch.org/whl/cu126 \
    --no-cache-dir \
    torch \
    torchvision \
&& micromamba run -n base \
  pip install \
    torchinfo

# Install necessary packages
USER root
RUN apt update && apt install -y \
  make
RUN apt upgrade -y && apt clean && rm -rf /var/lib/apt/lists/*

#===============================================================================
FROM build AS dev

# Remove default user
RUN if [ -n "$(id -u ubuntu)" ]; then userdel -r ubuntu; fi

# Define arguments
ARG USER_NAME
ARG USER_UID
ARG USER_GID

# Install necessary packages
RUN apt update && apt install -y \
  sudo \
  zsh \
  fzf \
  xclip \
  xsel \
  curl \
  wget \
  git \
  tree \
  tmux \
  vim
RUN apt upgrade -y && apt clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -s /bin/zsh --uid ${USER_UID} --gid mambauser -m ${USER_NAME}
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Change to non-root user
USER ${USER_NAME}

# Setup zsh shell with Oh-My-Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Activate micromamba in zsh
RUN echo \
  'eval "$(micromamba shell hook zsh)" && micromamba activate' \
  >> ~/.zshrc

ENTRYPOINT [ "zsh" ]
