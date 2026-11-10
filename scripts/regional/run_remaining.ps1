# Remaining stages of Paper C, in order. Finished jobs are skipped, so the script can be
# stopped (close the window or Ctrl+C) and started again at any time. Runs at idle priority.
#
#   cd C:\Users\leona\OneDrive\Desktop\MS_Research_Demonstrator\Application
#   powershell -ExecutionPolicy Bypass -File scripts\regional\run_remaining.ps1
#
# Progress goes to regional_fixes.log; one "stage <name> exit <code>" line per finished stage.
param([int]$Workers = 3)
$stages = "report_bench", "report", "bench_lse", "ablation", "truth_lse"
foreach ($s in $stages) {
    python -m scripts.regional.run --stage $s --workers $Workers >> regional_fixes.log 2>> regional_fixes.err
    "stage $s exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 regional_fixes.log
}
