#!/usr/bin/env bash
# Sequenciador do sweep PHerc1218 — roda janelas em série, extrai o
# essencial de cada log e acumula relatórios num único sumário.
#
# Uso:  ./sweep_band.sh Z_BEGIN1 Z_BEGIN2 ...
#       (cada janela = [Z, Z+800); ex.: ./sweep_band.sh 3940 3300 2660 2020 1380)
#
# Config por env (defaults do GATE2): PACK=7116a75, WIN=800.
# Saída: ~/challenges/vesuvius/fit1218/sweep_summary.txt (append)
# Interrompe a série se um run sair com exit != 0.

set -u
PACK="${PACK:-7116a75}"
WIN="${WIN:-800}"
# REPO: the vesuvius-sheet-tools checkout (default: two dirs up from this script)
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
# WORKDIR: where logs + the accumulated summary land
WORKDIR="${WORKDIR:-$REPO/spiral_work}"
FIT="$WORKDIR"
SUM="$FIT/sweep_summary.txt"

cd "$REPO" || exit 1

for ZB in "$@"; do
  ZE=$((ZB + WIN))
  TAG="z${ZB}-${ZE}"
  LOG="$FIT/logs/run_${TAG}.log"
  mkdir -p "$FIT/logs"

  echo "=================================================================" | tee -a "$SUM"
  echo "== JANELA ${ZB}-${ZE}  ($(date '+%d/%m %H:%M'))  pack@${PACK}" | tee -a "$SUM"

  FIT_PACK_REF="$PACK" FIT_Z_BEGIN="$ZB" FIT_Z_END="$ZE" \
    python reproduce/spiral_fit_window.py > "$LOG" 2>&1
  RC=$?

  # extrai o essencial do log
  {
    grep -E "exit |patches fetched|fitting [0-9]+ patch|unattached$|, [0-9]+ unattached" "$LOG" | tail -4
    grep -E "^step 29[468]00" "$LOG"
    grep -E "^satisfied_|^boundary_|WARNING: .* winding index" "$LOG"
    grep -cE "disconnected subrow" "$LOG" | sed 's/^/warnings_fragmentacao: /'
  } >> "$SUM"

  if [ $RC -ne 0 ]; then
    echo "!! EXIT $RC — série interrompida (log completo: $LOG)" | tee -a "$SUM"
    exit $RC
  fi

  OUT=$(ls -t "$REPO/spiral_work/out" | head -1)
  echo "-- window_report:" >> "$SUM"
  python3 "$(dirname "$0")/window_report.py" "$REPO/spiral_work/out/$OUT" >> "$SUM" 2>&1
  echo "" >> "$SUM"
done

echo "== SÉRIE COMPLETA ($(date '+%d/%m %H:%M')) ==" | tee -a "$SUM"
echo "Sumário em: $SUM"
