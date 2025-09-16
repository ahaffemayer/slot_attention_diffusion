#!/bin/sh
set -e
uv run meshcat-server &
exec uv run examples/main_shelf_slot_attention.py
