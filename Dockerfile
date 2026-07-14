# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    DOCLING_ARTIFACTS_PATH=/opt/docling-models

# PIP_TRUSTED_HOST env-var only accepts one host; pip.conf is the correct way
# to list multiple trusted hosts for Zscaler TLS interception environments.
# download.pytorch.org is included for the CPU-only torch pre-install step.
RUN printf '[global]\ntimeout = 600\ntrusted-host =\n\tpypi.org\n\tfiles.pythonhosted.org\n\tpypi.python.org\n\tdownload.pytorch.org\n\tdownload-r2.pytorch.org\n' \
    > /etc/pip.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Optionally install an organisation CA without copying it into the build
# context or committing it to source control.  Corporate builds should pass:
#   docker build --secret id=corporate_ca,src=/path/to/company-ca.crt ...
# Public-network builds can omit the secret.
RUN --mount=type=secret,id=corporate_ca,required=false \
    if [ -s /run/secrets/corporate_ca ]; then \
        install -m 0644 /run/secrets/corporate_ca \
            /usr/local/share/ca-certificates/corporate-ca.crt; \
        update-ca-certificates; \
    fi

COPY pyproject.toml .
COPY src/ src/
# On linux/aarch64 (Apple Silicon Docker), PyPI's torch wheel is already
# CPU-only — no separate pre-install step needed.
# On linux/amd64 the torch wheel from PyPI includes CUDA (~2 GB); the extra
# index steers pip to the smaller CPU-only build without overriding PyPI
# entirely (--extra-index-url, not --index-url, so aarch64 still resolves).
RUN pip install --prefer-binary \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -e . \
      fastapi uvicorn[standard] python-multipart

# Fetch Docling artifacts while the build has network access.  Runtime is
# explicitly offline so document processing never attempts a Hub download.
# The download is best-effort: in air-gapped / Zscaler environments it will
# fail; the container still starts and will use models mounted via the
# ./docling-models bind-mount in docker-compose.yml.  Run
#   make download-models
# on the host first so the bind-mount is populated before starting the stack.
RUN mkdir -p "${DOCLING_ARTIFACTS_PATH}" \
    && docling-tools models download -o "${DOCLING_ARTIFACTS_PATH}" \
    || echo "WARNING: Docling model download failed — mount ./docling-models at runtime"

ENV HF_HUB_OFFLINE=1

COPY . .

RUN mkdir -p data/input outputs/reports outputs/extracted_text outputs/rendered_pages

EXPOSE 8000

CMD ["python", "-m", "equation_extraction_pipeline.cli", "--batch"]
