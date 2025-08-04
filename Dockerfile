# --- Set up base image TAG (dynamic build) ---
ARG IMAGE_TAG
FROM cnstark/pytorch:${IMAGE_TAG}

# --- Set version variables (for use in images) ---
ARG MOIRAI_VERSION
ENV MOIRAI_VERSION=${MOIRAI_VERSION}

# --- Copy and install ---
COPY . /tmp/moirai

RUN set -eux; \
    cd /tmp/moirai && \
    pip install --upgrade pip && \
    pip install hatch && \
    rm -rf *.egg-info .eggs build dist && \
    hatch build && \
    pip install dist/*.whl && \
    rm -rf /tmp/moirai