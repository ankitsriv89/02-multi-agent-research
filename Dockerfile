# ─── Hugging Face Spaces (Docker SDK) container ──────────────────────────────
# HF Spaces conventions: expose 7860, run as a non-root user with UID 1000,
# write writable state under /home/user. Secrets (GROQ_API_KEY etc.) are
# injected by HF as environment variables — pydantic-settings reads them.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# lxml + readability-lxml need libxml2/libxslt at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxml2 libxslt1.1 gcc libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user named "user" with UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# Strip the build toolchain we needed for lxml.
USER root
RUN apt-get purge -y gcc libxml2-dev libxslt1-dev && apt-get autoremove -y
USER user

COPY --chown=user . .

# HF Spaces sends traffic to 7860.
ENV PORT=7860
EXPOSE 7860

# Streamlit needs --server.address=0.0.0.0 to be reachable inside the
# Spaces network, and --server.enableCORS=false because HF wraps the iframe.
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
