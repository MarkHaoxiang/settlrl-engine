# settlrl-render, self-contained: the frontend compiled in a node stage, the
# server installed from the uv workspace (CPU JAX). Run single-process — the
# game registry is in-memory, so extra workers would split games.
#
#   docker build -t settlrl-render .
#   docker run -p 8000:8000 -e SETTLRL_RENDER_CREATE_KEY=<secret> settlrl-render

FROM node:22-slim AS frontend
# The URL prefix the app is served under (assets, routes, API calls); pair a
# non-root value with ROOT_PATH at runtime, e.g. BASE_PATH=/settlrl/ behind a
# proxy that strips /settlrl.
ARG BASE_PATH=/
WORKDIR /build
COPY packages/settlrl-render/frontend/package.json packages/settlrl-render/frontend/package-lock.json ./
RUN npm ci
COPY packages/settlrl-render/frontend/ ./
RUN npm run build -- --base=$BASE_PATH

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
# No default groups: dev tooling and the CUDA jaxlib stay out of the image.
RUN uv sync --frozen --no-default-groups --package settlrl-render
COPY --from=frontend /build/dist packages/settlrl-render/frontend/dist
ENV RELOAD=0
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "settlrl-render"]
