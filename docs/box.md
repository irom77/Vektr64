# The box: things that bite

Facts about the play machine `irom-sei` that are easy to trip over. Add a line here whenever
one costs time. How to operate the bot is in `docs/runbook.md`; this file is only the traps.

## Access

- SSH over Tailscale: `ssh irom@100.114.82.19`. The repository is `/home/irom/Vektr64`.
- **No desktop runs** (since 2026-11-28): `gdm` is stopped and the boot default is
  `multi-user.target`. The idle login screen alone held 2.7 GB of the 7.8 GB of memory.
  A monitor and keyboard give a text login. `sudo systemctl start gdm` brings the desktop
  back until the next reboot; `sudo systemctl set-default graphical.target` makes that permanent.
- **Wi-Fi does not need a login.** The connection `jablko` is defined system-wide in
  `/etc/netplan/` with its key, and connects at boot. `grucha` (Wi-Fi) and the wired port
  `enp1s0` also connect by themselves. Plugging in a cable is the fallback if Wi-Fi fails.
- The box is administered over the same Wi-Fi link it plays on. Never restart networking,
  Tailscale or the firewall remotely without a second way in.
- The box's clock is US Eastern time; `journalctl` shows local time, the store uses UTC.

## It is a shared machine

- Vektr64 is a guest. Also running: the Hermes personal agent (about 460 MB), Kasm Workspaces
  in Docker (about 1.1 GB), Home Assistant and CouchDB. Do not stop, restart or reconfigure them.
- 4 physical cores, 8 threads, 7.8 GB of memory. Swap is normally in use (2 GB or more); that
  alone is not a fault. Look at `available` in `free -m` and at `/proc/pressure/`.
- Background work is held down by `vektr64-worker.service`: `Nice=19`, `MemoryMax=700M`,
  2 games at a time. Anything new that runs in the worker must fit in that memory limit,
  including engine builds.
- The engine is slower here than on the laptop: about 2.85M positions per second against
  4.4–5.5M. Timings measured on the laptop do not carry over.

## Files and commands

- **Never `source .env`** in a shell. The file has Windows line endings and a mistake prints a
  token. Read the store through the ops code:
  `uv run --project ops python -c "from pathlib import Path; from vektr_ops.cli import the_store; s = the_store(Path('.')); …"`.
- **Game record file names contain spaces** (`bot/game_records/octarine-bot vs Vektr64 - ua63qkWw.pgn`).
  A shell `for f in *.pgn` loop over SSH breaks; use Python or quote everything.
- `bot/config.yml` is generated from `bot/config.template.yml` at every start. Edit the template.
- What plays is `builds/current`, set by the ten-minute sync from the manifest, not the checked-out
  commit. Before `git pull`, check that `vektr-ops status` shows a pinned champion.
- **`vektr-ops gate` promotes by itself on `accept`**, and the box switches within ten minutes.
  Starting a gate is therefore a decision to change what plays if the candidate wins.
- `sudo systemctl restart vektr64` abandons the game in progress.

## Checking health without changing anything

```bash
systemctl is-active vektr64 vektr64-worker vektr64-page vektr64-sync.timer
free -m; cat /proc/pressure/cpu /proc/pressure/memory
grep -l "Time forfeit" bot/game_records/*.pgn      # should print nothing
uv run --project ops vektr-ops status
```
