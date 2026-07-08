#!/bin/sh
set -eu

ROOT_DIR="${ROOT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
THESIS_ROOT="${THESIS_ROOT:-$ROOT_DIR/thesis/thesis}"
SOURCE_ROOT="${SOURCE_ROOT:-$THESIS_ROOT/assets/results}"
REPORT_FIGURES_DIR="${REPORT_FIGURES_DIR:-$ROOT_DIR/reports/tfg_figures}"
STRICT="${STRICT:-1}"
INCLUDE_README="${INCLUDE_README:-1}"

TMP_PARENT="${TMPDIR:-/tmp}"
TMP_DIR="$(mktemp -d "$TMP_PARENT/tfg-figures.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT HUP INT TERM

REFERENCES="$TMP_DIR/references.tsv"
COPIED="$TMP_DIR/copied.txt"
MISSING="$TMP_DIR/missing.txt"
: > "$REFERENCES"
: > "$COPIED"
: > "$MISSING"

csv_escape() {
  printf '%s' "$1" | sed 's/"/""/g'
}

repo_relative() {
  case "$1" in
    "$ROOT_DIR"/*) printf '%s\n' "${1#"$ROOT_DIR"/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

add_reference() {
  figure="$1"
  used_by="$2"
  if [ -n "$figure" ]; then
    printf '%s\t%s\n' "$figure" "$used_by" >> "$REFERENCES"
  fi
}

collect_tex_file() {
  file="$1"
  used_by="$(repo_relative "$file")"
  matches="$(grep -Eo '\\(safeincludegraphics|includegraphics)(\[[^]]*\])?\{results/[^}]+\}' "$file" || true)"
  if [ -z "$matches" ]; then
    return 0
  fi
  printf '%s\n' "$matches" \
    | sed -E 's#.*\{results/([^}]+)\}#\1#' \
    | while IFS= read -r figure; do
        add_reference "$figure" "$used_by"
      done
}

collect_readme() {
  readme="$ROOT_DIR/README.md"
  if [ "$INCLUDE_README" != "1" ] || [ ! -f "$readme" ]; then
    return 0
  fi
  matches="$(grep -Eo 'thesis/thesis/assets/results/[^"<>[:space:]]+' "$readme" || true)"
  if [ -z "$matches" ]; then
    return 0
  fi
  printf '%s\n' "$matches" \
    | sed 's#^thesis/thesis/assets/results/##' \
    | while IFS= read -r figure; do
        add_reference "$figure" "README.md"
      done
}

mkdir -p "$REPORT_FIGURES_DIR/results"

find "$THESIS_ROOT" -type f -name '*.tex' | sort | while IFS= read -r file; do
  collect_tex_file "$file"
done
collect_readme

MANIFEST="$REPORT_FIGURES_DIR/manifest.csv"
MISSING_REPORT="$REPORT_FIGURES_DIR/missing_figures.txt"
printf 'figure,source,destination,used_by\n' > "$MANIFEST"
: > "$MISSING_REPORT"

if [ ! -s "$REFERENCES" ]; then
  printf '[WARN] No thesis/README result figures were found.\n'
else
  cut -f 1 "$REFERENCES" | sort -u | while IFS= read -r figure; do
    source="$SOURCE_ROOT/$figure"
    destination="$REPORT_FIGURES_DIR/results/$figure"
    used_by="$(
      awk -F '\t' -v figure="$figure" '$1 == figure { print $2 }' "$REFERENCES" \
        | sort -u \
        | paste -sd '|' -
    )"

    if [ -f "$source" ]; then
      mkdir -p "$(dirname "$destination")"
      cp -p "$source" "$destination"
      printf '%s\n' "$figure" >> "$COPIED"
      printf '"%s","%s","%s","%s"\n' \
        "$(csv_escape "$figure")" \
        "$(csv_escape "$source")" \
        "$(csv_escape "$destination")" \
        "$(csv_escape "$used_by")" \
        >> "$MANIFEST"
    else
      printf '%s\n' "$figure" >> "$MISSING"
      printf '%s\n' "$figure" >> "$MISSING_REPORT"
    fi
  done
fi

copied_count="$(wc -l < "$COPIED" | tr -d ' ')"
missing_count="$(wc -l < "$MISSING" | tr -d ' ')"

printf '[OK] Copied %s TFG figures into: %s\n' "$copied_count" "$REPORT_FIGURES_DIR/results"
if [ "$missing_count" != "0" ]; then
  printf '[ERROR] Missing %s referenced figures. See: %s\n' "$missing_count" "$MISSING_REPORT"
  if [ "$STRICT" = "1" ]; then
    exit 1
  fi
fi
