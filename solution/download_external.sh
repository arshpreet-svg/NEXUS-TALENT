#!/bin/sh
# Downloads the public Metrica Sports sample games 1-3 (other matches) into data/external/.
# Source: https://github.com/metrica-sports/sample-data (anonymised sample data; please credit Metrica Sports).
set -e
cd "$(dirname "$0")/.."
mkdir -p data/external && cd data/external
B=https://raw.githubusercontent.com/metrica-sports/sample-data/master/data
for g in 1 2; do
  for f in RawEventsData RawTrackingData_Home_Team RawTrackingData_Away_Team; do
    [ -f "Sample_Game_${g}_$f.csv" ] || curl -sSfO "$B/Sample_Game_$g/Sample_Game_${g}_$f.csv"
  done
done
for f in Sample_Game_3_events.json Sample_Game_3_tracking.txt; do
  [ -f "$f" ] || curl -sSfO "$B/Sample_Game_3/$f"
done
