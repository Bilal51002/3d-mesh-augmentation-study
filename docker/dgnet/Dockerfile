# Environnement CPU officiel de Jittor
FROM jittor/jittor

WORKDIR /workspace/dgnet

# Dépendances système nécessaires à DGNet/Jittor
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Dépendances officielles du dépôt DGNet
COPY models/dgnet/requirements.txt /tmp/dgnet-requirements.txt

# Jittor est déjà installé dans l'image officielle.
# On retire donc "jittor" du requirements avant d'installer le reste.
RUN sed '/^[[:space:]]*jittor[[:space:]]*$/Id' \
        /tmp/dgnet-requirements.txt \
        > /tmp/dgnet-requirements-no-jittor.txt \
    && python3.7 -m pip install --no-cache-dir \
        "pip<24.1" \
        "setuptools<69" \
        wheel \
    && python3.7 -m pip install --no-cache-dir \
        -r /tmp/dgnet-requirements-no-jittor.txt \
    && python3.7 -m pip install --no-cache-dir \
        "numpy<1.22" \
        "scipy<1.8" \
        "pandas<1.4" \
        "scikit-learn<1.1" \
        "matplotlib<3.6" \
        "psutil<6"

ENV PYTHONPATH=/workspace/dgnet

CMD ["/bin/bash"]
