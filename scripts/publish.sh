#!/usr/bin/env bash
# Veroeffentlicht EINE Pipeline in R2: metadata mergen -> gzip -> Binaerdatei -> metadata -> Altlasten loeschen
# usage: publish.sh <name> <meta_script> <out_dir> <file_prefix>
set -euo pipefail
NAME="$1"; META_SCRIPT="$2"; OUT_DIR="$3"; FILE_PREFIX="$4"

F="${OUT_DIR}/${FILE_PREFIX}_${BERLIN_DATE}_${BERLIN_TIME}"
if [ ! -f "$F" ]; then
  echo "::error::[$NAME] Ausgabedatei $F fehlt - nichts zu veroeffentlichen."
  exit 1
fi

WORK="${RUNNER_TEMP}/publish_${NAME}"; mkdir -p "$WORK"
EP=(--endpoint-url "$R2_ENDPOINT")
S3="s3://${R2_BUCKET}/${NAME}"

aws s3 cp "${S3}/metadata.json" "$WORK/metadata_prev.json" "${EP[@]}" \
  || echo "[$NAME] Keine bestehende metadata.json in R2 - starte frisch."

# OUT_DIR statt "output": jede Pipeline sieht nur ihre eigenen Dateien
python "$META_SCRIPT" "$OUT_DIR" \
  --run "$RUN" --date "$BERLIN_DATE" \
  --existing "$WORK/metadata_prev.json" --max-keep "$MAX_KEEP" \
  --meta-out "$WORK/metadata.json" --removed-out "$WORK/removed_files.txt"

gzip -9 -n -c "$F" > "$F.tmp" && mv "$F.tmp" "$F"

aws s3 cp "$F" "${S3}/$(basename "$F")" --content-type application/gzip "${EP[@]}"
# metadata erst NACH der Binaerdatei, damit Clients nie auf fehlende Dateien verweisen
aws s3 cp "$WORK/metadata.json" "${S3}/metadata.json" "${EP[@]}"

if [ -s "$WORK/removed_files.txt" ]; then
  while IFS= read -r fname || [ -n "$fname" ]; do
    [ -z "$fname" ] && continue
    echo "[$NAME] Loesche $fname aus R2 ..."
    aws s3 rm "${S3}/$fname" "${EP[@]}"
  done < "$WORK/removed_files.txt"
fi
echo "[$NAME] fertig."