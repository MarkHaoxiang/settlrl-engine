# settlrl-app, self-contained: the frontend compiled in a node stage, the
# server installed from the uv workspace (CPU JAX). Run single-process — the
# game registry is in-memory, so extra workers would split games.
#
#   docker build -t settlrl-app .
#   docker run -p 8000:8000 settlrl-app

FROM node:22-slim AS frontend
# The URL prefix the app is served under (assets, routes, API calls); pair a
# non-root value with ROOT_PATH at runtime, e.g. BASE_PATH=/settlrl/ behind a
# proxy that strips /settlrl.
ARG BASE_PATH=/
WORKDIR /build
COPY packages/settlrl-app/frontend/package.json packages/settlrl-app/frontend/package-lock.json ./
RUN npm ci
COPY packages/settlrl-app/frontend/ ./
RUN npm run build -- --base=$BASE_PATH

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
# No default groups: dev tooling and the CUDA jaxlib stay out of the image.
RUN uv sync --frozen --no-default-groups --package settlrl-app
COPY --from=frontend /build/dist packages/settlrl-app/frontend/dist
ENV RELOAD=0
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "settlrl-app"]
