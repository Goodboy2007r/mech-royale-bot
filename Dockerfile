# The match runs it with no network, a read-only root (writable /tmp), 1 CPU, 512 MB, and the map at /arena.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /bot
COPY bot.py world.py mechfight.py ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
