param([string]$PythonPath, [ValidateRange(1, 4)][int]$Workers = 2)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) { $PythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Project Python environment missing; install requirements-lock.txt first' }
$pipelinePath = Join-Path $projectRoot 'daily_pipeline.py'
if (-not (Test-Path -LiteralPath $pipelinePath)) { throw 'Pipeline script missing' }
$indiaZone = [System.TimeZoneInfo]::FindSystemTimeZoneById('India Standard Time')
$indiaNow = [System.TimeZoneInfo]::ConvertTimeFromUtc([datetime]::UtcNow, $indiaZone)
$indiaRun = [datetime]::SpecifyKind($indiaNow.Date.AddHours(18).AddMinutes(30), [DateTimeKind]::Unspecified)
$localRun = [System.TimeZoneInfo]::ConvertTime($indiaRun, $indiaZone, [System.TimeZoneInfo]::Local)
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $localRun
$action = New-ScheduledTaskAction -Execute $PythonPath -Argument ('-u "' + $pipelinePath + '" --workers ' + $Workers) -WorkingDirectory $projectRoot
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 4) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$userName = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $userName -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName 'Equimaster-Daily-Refresh' -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Equimaster market, model and news refresh at 18:30 India time on weekdays; runs while this user is logged in.' -Force | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName 'Equimaster-Daily-Refresh' | Select-Object NextRunTime,LastTaskResult
$duplicate = Get-ScheduledTask -TaskName 'Equimaster daily update' -ErrorAction SilentlyContinue
if ($duplicate -and $duplicate.State -ne 'Disabled') {
    try { Disable-ScheduledTask -TaskName $duplicate.TaskName -ErrorAction Stop | Select-Object TaskName,State }
    catch { Write-Warning 'Windows denied disabling Equimaster daily update. Run this installer once as administrator; the shared pipeline lock prevents overlapping runs meanwhile.' }
}
