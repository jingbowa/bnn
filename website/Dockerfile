FROM node:22-alpine AS build
WORKDIR /app
COPY . .
RUN node build.mjs
FROM node:22-alpine
WORKDIR /app
COPY --from=build --chown=node:node /app/dist ./dist
COPY --chown=node:node server.mjs ./
USER node
ENV PORT=8080
EXPOSE 8080
CMD ["node", "server.mjs"]
