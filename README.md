# Mech Royale bot

A Python bot for the Builders Buildathon 2026 Mech Royale track (protocol `mechfight/1`). It uses only the standard library.

Files:

- `bot.py`: the bot (strategy, movement, aiming, weapon and upgrade choices)
- `world.py`: map loader, terrain sampling, prism line-of-sight, a walkability grid with A* and cover search
- `mechfight.py`: the kit's stdin/stdout wrapper, unchanged
- `Dockerfile`: `python:3.13-slim`, entrypoint `python3 -u /bot/bot.py`

What it does each decision:

1. Reads the zone and moves inside the next circle before it shrinks, boosting when it is late.
2. Fights the mech it can hit: leads gun shots, charges and releases the laser on target, fires lightning inside its cone, lobs mortar shells over cover when the gun has no line of fire.
3. Breaks line of sight when several enemies are in range, choosing a nearby point that the fewest of them can see.
4. Loots when nothing threatens it: lasers first, cubes, crates and drones (which it shoots), and supply drops it can hold alone.
5. Dodges incoming shells and bullets, and steps out of the line of a charging laser.

Every number comes from `hello.config`, so balance changes do not need code changes. A* has a 12 ms time budget per search so a decision stays well inside the 50 ms limit.

Check it locally with the kit:

```sh
python3 check/check.py -- python3 -u bot.py
docker build -t my-bot .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" my-bot
```
