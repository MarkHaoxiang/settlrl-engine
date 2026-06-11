# catan-render, self-contained: the frontend compiled in a node stage, the
# server installed from the uv workspace (CPU JAX). Run single-process — the
# game registry is in-memory, so extra workers would split games.
#
#   docker build -t catan-render .
#   docker run -p 8000:8000 -e CATAN_RENDER_CREATE_KEY=<secret> catan-render

FROM node:22-slim AS frontend
WORKDIR /build
COPY packages/catan-render/frontend/package.json packages/catan-render/frontend/package-lock.json ./
RUN npm ci
COPY packages/catan-render/frontend/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
# No default groups: dev tooling and the CUDA jaxlib stay out of the image.
RUN uv sync --frozen --no-default-groups --package catan-render
COPY --from=frontend /build/dist packages/catan-render/frontend/dist
ENV RELOAD=0
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "catan-render"]
