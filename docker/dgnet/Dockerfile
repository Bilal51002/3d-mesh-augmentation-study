# Environnement CPU officiel de Jittor
FROM jittor/jittor

# Dossier de travail par défaut
WORKDIR /workspace/dgnet

# Dépendances système nécessaires à DGNet et Jittor
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances officielles de DGNet
COPY models/dgnet/requirements.txt /tmp/dgnet-requirements.txt

# Jittor est déjà installé dans l'image officielle.
# Installation des autres dépendances compatibles avec Python 3.7.
RUN sed '/^[[:space:]]*jittor[[:space:]]*$/Id' \
        /tmp/dgnet-requirements.txt \
        > /tmp/dgnet-requirements-no-jittor.txt \
    && python3.7 -m pip install --no-cache-dir \
        "pip<24.1" \
        "setuptools<69" \
        wheel \
    && python3.7 -m pip install --no-cache-dir \
        -r /tmp/dgnet-requirements-no-jittor.txt \
    && python3.7 -m pip install --no-cache-dir --upgrade \
        "numpy==1.21.6" \
        "trimesh==3.23.5" \
        "scipy<1.8" \
        "pandas<1.4" \
        "scikit-learn<1.1" \
        "matplotlib<3.6" \
        "psutil<6" \
        "tensorboardX==2.5.1"

# Rendre le dépôt officiel DGNet importable
ENV PYTHONPATH=/workspace/dgnet

# Commande exécutée par défaut
CMD ["/bin/bash"]
