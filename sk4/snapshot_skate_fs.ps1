<#
.SYNOPSIS
  Snapshot (or diff) the files skate. can write to - the Windows twin of tools/snapshot_skate_fs.py.

.DESCRIPTION
  Take a snapshot BEFORE updating skate. / downloading high-quality textures, another AFTER, then
  diff them. Anything NEW or CHANGED is what the update or the game downloaded. The JSON is the
  same shape the Linux script writes - keys are paths relative to the drive root with forward
  slashes, values are [size, unix mtime] - so either script can diff snapshots from the other.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\snapshot_skate_fs.ps1 snap skate_fs_before.json
  powershell -ExecutionPolicy Bypass -File tools\snapshot_skate_fs.ps1 snap skate_fs_after.json -Extra 'D:\skate-hq-textures'
  powershell -ExecutionPolicy Bypass -File tools\snapshot_skate_fs.ps1 diff skate_fs_before.json skate_fs_after.json > skate_fs_diff.txt
#>
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet('snap', 'diff')][string]$Mode,
    [Parameter(Mandatory = $true, Position = 1)][string]$First,
    [Parameter(Position = 2)][string]$Second,
    [string]$SteamRoot = 'C:\Program Files (x86)\Steam',
    [string]$UserProfile = $env:USERPROFILE,
    [string[]]$Extra = @()
)

function Get-Key([string]$full) {
    $root = [System.IO.Path]::GetPathRoot($full)
    return $full.Substring($root.Length).Replace('\', '/')
}

if ($Mode -eq 'snap') {
    $roots = @(
        (Join-Path $SteamRoot 'steamapps\common\Skate'),
        (Join-Path $SteamRoot 'steamapps\appmanifest_3354750.acf'),
        (Join-Path $UserProfile 'AppData\Local\Skate')
    ) + $Extra
    $files = [ordered]@{}
    foreach ($r in $roots) {
        if (-not (Test-Path -LiteralPath $r)) { Write-Warning "missing: $r"; continue }
        $items = if ((Get-Item -LiteralPath $r).PSIsContainer) {
            Get-ChildItem -LiteralPath $r -Recurse -File -Force -ErrorAction SilentlyContinue
        } else { @(Get-Item -LiteralPath $r) }
        foreach ($f in $items) {
            $mt = [int64]([DateTimeOffset]$f.LastWriteTimeUtc).ToUnixTimeSeconds()
            $files[(Get-Key $f.FullName)] = @([int64]$f.Length, $mt)
        }
    }
    $files | ConvertTo-Json -Depth 3 -Compress | Set-Content -LiteralPath $First -Encoding UTF8
    $gb = ($files.Values | ForEach-Object { $_[0] } | Measure-Object -Sum).Sum / 1e9
    Write-Output ("{0:N0} files, {1:N2} GB -> {2}" -f $files.Count, $gb, $First)
    exit 0
}

if (-not $Second) { Write-Error 'diff needs two snapshot files'; exit 2 }
function Read-Snap([string]$p) {
    $o = Get-Content -LiteralPath $p -Raw | ConvertFrom-Json
    $h = @{}
    foreach ($prop in $o.PSObject.Properties) { $h[$prop.Name] = @([int64]$prop.Value[0], [int64]$prop.Value[1]) }
    return $h
}
$A = Read-Snap $First
$B = Read-Snap $Second
$new = @($B.Keys | Where-Object { -not $A.ContainsKey($_) })
$gone = @($A.Keys | Where-Object { -not $B.ContainsKey($_) })
$changed = @($B.Keys | Where-Object { $A.ContainsKey($_) -and (($A[$_][0] -ne $B[$_][0]) -or ($A[$_][1] -ne $B[$_][1])) })
foreach ($set in @(@('NEW', $new, $B), @('CHANGED', $changed, $B), @('GONE', $gone, $A))) {
    $label, $keys, $src = $set
    $mb = ($keys | ForEach-Object { $src[$_][0] } | Measure-Object -Sum).Sum / 1e6
    Write-Output ("{0}: {1} files, {2:N1} MB" -f $label, $keys.Count, $mb)
    $keys | Sort-Object { - $src[$_][0] } | ForEach-Object {
        $extra = if ($label -eq 'CHANGED') { "  (was $($A[$_][0]))" } else { '' }
        Write-Output ("  {0,14:N0}  {1}{2}" -f $src[$_][0], $_, $extra)
    }
}
