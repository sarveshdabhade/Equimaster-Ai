param([string]$PythonPath = (Get-Command python -ErrorAction Stop).Source)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pipelinePath = Join-Path $projectRoot 'daily_pipeline.py'
if (-not (Test-Path -LiteralPath $pipelinePath)) { throw 'Pipeline script missing' }
$indiaZone = [System.TimeZoneInfo]::FindSystemTimeZoneById('India Standard Time')
$indiaNow = [System.TimeZoneInfo]::ConvertTimeFromUtc([datetime]::UtcNow, $indiaZone)
$indiaRun = [datetime]::SpecifyKind($indiaNow.Date.AddHours(18).AddMinutes(30), [DateTimeKind]::Unspecified)
$localRun = [System.TimeZoneInfo]::ConvertTime($indiaRun, $indiaZone, [System.TimeZoneInfo]::Local)
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $localRun
$action = New-ScheduledTaskAction -Execute $PythonPath -Argument ('-u "' + $pipelinePath + '"') -WorkingDirectory $projectRoot
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 4)
$userName = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $userName -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName 'Equimaster-Daily-Refresh' -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Equimaster market, model and news refresh at 18:30 India time on weekdays; runs while this user is logged in.' -Force | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName 'Equimaster-Daily-Refresh' | Select-Object NextRunTime,LastTaskResult
