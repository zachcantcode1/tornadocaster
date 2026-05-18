# deploy_build.ps1 - Launch a dataset build on a DigitalOcean droplet and
# download the resulting Parquet shards when done.
#
# Usage:
#   .\scripts\deploy_build.ps1 -DropletIP 123.45.67.89
#   .\scripts\deploy_build.ps1 -DropletIP 123.45.67.89 -Years "2021 2022 2023"
#   .\scripts\deploy_build.ps1 -DropletIP 123.45.67.89 -DownloadOnly

param(
    [Parameter(Mandatory=$true)]
    [string]$DropletIP,

    [string]$Years = "2021 2022 2023",

    [string]$RemoteDir = "/root/tornadocaster",

    [string]$LocalOutDir = "data\training",

    [switch]$DownloadOnly
)

$ErrorActionPreference = "Stop"
$SSHTarget = "root@$DropletIP"
$RemoteOut = "$RemoteDir/data/training"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Tornadocaster Remote Build" -ForegroundColor Cyan
Write-Host "  Droplet : $DropletIP" -ForegroundColor Cyan
Write-Host "  Years   : $Years" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

if (-not $DownloadOnly) {
    Write-Host "`n[1/3] Uploading bootstrap script..." -ForegroundColor Yellow
    scp -o StrictHostKeyChecking=accept-new `
        scripts\remote_bootstrap.sh `
        "${SSHTarget}:/root/remote_bootstrap.sh"

    Write-Host "[2/3] Running bootstrap on droplet (3-5 min for setup)..." -ForegroundColor Yellow
    ssh -o StrictHostKeyChecking=accept-new $SSHTarget "YEARS='$Years' bash /root/remote_bootstrap.sh"

    Write-Host "`nBuild is running remotely in tmux session 'build'." -ForegroundColor Green
    Write-Host "SSH in to watch: ssh $SSHTarget  then: tmux attach -t build" -ForegroundColor Gray
    Write-Host ""

    Write-Host "[3/3] Waiting for build to complete (polling every 60s)..." -ForegroundColor Yellow
    Write-Host "      Press Ctrl+C to stop waiting and download manually later." -ForegroundColor Gray
    Write-Host ""

    $done = $false
    $polls = 0
    while (-not $done) {
        Start-Sleep -Seconds 60
        $polls++
        $status = ssh $SSHTarget "tail -3 $RemoteDir/build.log 2>/dev/null || echo 'log not ready'"
        $shardCount = ssh $SSHTarget "ls $RemoteOut/*.parquet 2>/dev/null | wc -l || echo 0"
        Write-Host "  [$($polls)min] Shards: $($shardCount.Trim())  |  $($status[-1])" -ForegroundColor Gray

        if ($status -match "BUILD COMPLETE") {
            $done = $true
            Write-Host "`n  Build complete!" -ForegroundColor Green
        }
    }
}

Write-Host "`nDownloading Parquet shards..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $LocalOutDir | Out-Null

$rsyncAvail = $null -ne (Get-Command rsync -ErrorAction SilentlyContinue)
if ($rsyncAvail) {
    rsync -avz --progress "${SSHTarget}:${RemoteOut}/" "$LocalOutDir/"
} else {
    Write-Host "  rsync not found - using scp" -ForegroundColor Yellow
    scp -r "${SSHTarget}:${RemoteOut}/*.parquet" "$LocalOutDir\"
}

$shardFiles = Get-ChildItem "$LocalOutDir\*.parquet" -ErrorAction SilentlyContinue
$totalMB = [math]::Round(($shardFiles | Measure-Object Length -Sum).Sum / 1MB, 1)

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  Download complete!" -ForegroundColor Green
Write-Host "  Shards : $($shardFiles.Count) files" -ForegroundColor Green
Write-Host "  Size   : $totalMB MB" -ForegroundColor Green
Write-Host "  Path   : $((Resolve-Path $LocalOutDir).Path)" -ForegroundColor Green
Write-Host ""
Write-Host "  Next step - train the model locally:" -ForegroundColor Cyan
Write-Host "    python -m src.training.train --data data/training --out models/tornado_lgbm.pkl" -ForegroundColor White
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Remember to DESTROY the droplet when done!" -ForegroundColor Red
Write-Host "  https://cloud.digitalocean.com/droplets" -ForegroundColor Gray
