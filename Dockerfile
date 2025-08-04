# --- Set up base image TAG (dynamic build) ---
ARG IMAGE_TAG
FROM cnstark/pytorch:${IMAGE_TAG}

# --- Set version variables (for use in images) ---
ARG PIPEAI_VERSION
ENV PIPEAI_VERSION=${PIPEAI_VERSION}

# --- Copy and install ---
COPY . /tmp/pipeai

RUN set -eux; \
    cd /tmp/pipeai && \
    pip install --upgrade pip && \
    pip install hatch && \
    rm -rf *.egg-info .eggs build dist && \
    hatch build && \
    pip install dist/*.whl && \
    rm -rf /tmp/pipeai