#!/bin/bash
# Keep a NEAT run alive: restart from its checkpoint if it crashes (exit != 0), and log the crash as a site event.
# A clean stop (SIGTERM -> checkpoint -> exit 0) or a finished curriculum ends the supervisor too.
#   setsid nohup ./supervise.sh M1 curricula/M1.json [evo.train args] > logs/supervise_M1.log 2>&1 &
cd "$(dirname "$0")"
RUN=$1; CUR=$2; shift 2
crashes=0
while true; do
  .venv/bin/python -m evo.train --curriculum "$CUR" --run "$RUN" "$@" >> "logs/run_$RUN.log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then echo "$(date '+%F %T') clean exit; supervisor done"; exit 0; fi
  crashes=$((crashes + 1))
  err=$(grep -E "Error|assert" "logs/run_$RUN.log" | tail -1 | tr '"' "'" | cut -c1-200)
  echo "$(date '+%F %T') crashed (exit $code, crash $crashes): $err; restarting in 30 s"
  .venv/bin/python -c "import json,time,sys; open('runs/$RUN/events.jsonl','a').write(json.dumps({'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'kind': 'crash', 'stage_index': -1, 'note': sys.argv[1]})+'\n')" "exit $code: $err (restarted from the last checkpoint by supervise.sh)"
  if [ $crashes -ge 5 ]; then echo "5 crashes: giving up"; exit 1; fi
  sleep 30
done
