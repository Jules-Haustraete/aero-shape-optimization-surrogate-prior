# SU2 8.5.0 (RANS truth) + gmshairfoil2d (mesher) + the Python stack of the optimizer, on CPU.
# The optimizer needs no GPU; the thesis runs used a PyTorch 2.1 CUDA image on the EPFL RCP cluster.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends wget unzip libgomp1 \
      libglu1-mesa libxrender1 libxcursor1 libxinerama1 libxft2 libgl1 libfontconfig1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# SU2 v8.5.0, OpenMP build. The release zip is nested: unzip twice.
RUN wget -q https://github.com/su2code/SU2/releases/download/v8.5.0/SU2-v8.5.0-linux64-omp.zip -O /tmp/su2.zip \
 && mkdir -p /tmp/su2x && unzip -q /tmp/su2.zip -d /tmp/su2x \
 && unzip -q /tmp/su2x/*.zip -d /opt/su2 \
 && chmod +x /opt/su2/bin/SU2_* \
 && rm -rf /tmp/su2.zip /tmp/su2x

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Read by the code; OMP_NUM_THREADS must match the cores given to SU2 (see README)
ENV SU2_BIN=/opt/su2/bin/SU2_CFD GMSH_BIN=/usr/local/bin/gmshairfoil2d OMP_NUM_THREADS=4
WORKDIR /workspace
COPY . /workspace
CMD ["/bin/bash"]
