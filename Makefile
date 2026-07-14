build:
	docker compose build

up:
	docker compose up

down:
	docker compose down

install:
	.venv/bin/pip install -e ".[dev]"

run:
	.venv/bin/python -m equation_extraction_pipeline.cli --pdf $(PDF)

batch:
	.venv/bin/python -m equation_extraction_pipeline.cli --batch

test:
	.venv/bin/python -m pytest tests/ -v

lint:
	.venv/bin/ruff check src/ tests/
	.venv/bin/ruff format --check src/ tests/

# Download Docling layout/table/formula models into ./docling-models so the
# bind-mount in docker-compose.yml can serve them to the container.
# Run this once before `make up`.  The macOS System keychain (Zscaler root CA)
# is merged with certifi so HuggingFace TLS verification succeeds on
# corporate networks.
download-models:
	@echo "Building CA bundle (certifi + macOS keychains)..."
	@CERTIFI=$$(python3 -c "import certifi; print(certifi.where())"); \
	 CABUNDLE=$$(mktemp /tmp/combined-ca.XXXXXX.pem); \
	 cat "$$CERTIFI" > "$$CABUNDLE"; \
	 security export -t certs -f pemseq -k /Library/Keychains/System.keychain 2>/dev/null >> "$$CABUNDLE" || true; \
	 security export -t certs -f pemseq -k /System/Library/Keychains/SystemRootCertificates.keychain 2>/dev/null >> "$$CABUNDLE" || true; \
	 echo "Downloading Docling models..."; \
	 REQUESTS_CA_BUNDLE="$$CABUNDLE" CURL_CA_BUNDLE="$$CABUNDLE" \
	   docling-tools models download -o ./docling-models; \
	 STATUS=$$?; rm -f "$$CABUNDLE"; exit $$STATUS
