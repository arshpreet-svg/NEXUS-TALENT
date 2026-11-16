#!/bin/sh
# Downloads the DFL open Bundesliga dataset (7 matches, ~2.6 GB) into data/external/dfl/ and verifies MD5.
# Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event data in elite
# soccer", Scientific Data. figshare 10.6084/m9.figshare.28196177, licence CC BY 4.0.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/external/dfl && cd data/external/dfl

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }

curl -sSf -m 60 https://api.figshare.com/v2/articles/28196177 | python3 -c "
import sys, json
for f in json.load(sys.stdin)['files']:
    if 'positions_raw' in f['name'] or 'matchinformation' in f['name']:
        print(f['name'], f['download_url'], f['computed_md5'])
" > manifest.txt

fetch() {  # name url md5 : download (resumable) unless already correct
  if [ -f "$1" ] && [ "$(md5_of "$1")" = "$3" ]; then return 0; fi
  curl -sSfL -C - --retry 10 --retry-delay 3 -o "$1" "$2"
}
while read n u m; do fetch "$n" "$u" "$m" & done < manifest.txt
wait

bad=0
while read n u m; do
  [ "$(md5_of "$n")" = "$m" ] || { echo "CHECKSUM MISMATCH $n"; bad=1; }
done < manifest.txt
[ $bad = 0 ] && echo "DFL data OK"
exit $bad
