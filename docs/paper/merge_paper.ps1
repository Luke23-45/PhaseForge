# PowerShell script to merge all paper sections into a single Markdown file.
# If final_paper.md already exists, it creates final_paper_1.md, final_paper_2.md, etc.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$sectionFiles = @(
    "abstract/abstract.md",
    "1.introduction/introduction.md",
    "2.Related_Work/related_work.md",
    "3.method/method.md",
    "4.experimental_setup/experimental_setup.md",
    "5.Results/results.md",
    "6.discussion_limitations/discussion.md",
    "7.conclusion/conclusion.md"
)

$baseName = "final_paper"
$ext = ".md"
$targetPath = Join-Path $scriptDir "$baseName$ext"

if (Test-Path -Path $targetPath) {
    $counter = 1
    do {
        $candidateName = "${baseName}_${counter}${ext}"
        $targetPath = Join-Path $scriptDir $candidateName
        $counter++
    } while (Test-Path -Path $targetPath)
}

$mergedParts = [System.Collections.Generic.List[string]]::new()

foreach ($relPath in $sectionFiles) {
    $fullPath = Join-Path $scriptDir $relPath
    if (-not (Test-Path -Path $fullPath -PathType Leaf)) {
        Write-Error "Required section file not found: $fullPath"
        exit 1
    }
    $content = Get-Content -Path $fullPath -Raw -Encoding utf8
    $mergedParts.Add($content.Trim())
}

$divider = "`n`n---`n`n"
$finalDocument = ($mergedParts -join $divider) + "`n"

[System.IO.File]::WriteAllText($targetPath, $finalDocument, [System.Text.Encoding]::UTF8)

$lineCount = ($finalDocument -split "`r?`n").Count
$wordCount = ($finalDocument -split "\s+").Count

Write-Host "Successfully merged $($sectionFiles.Count) sections."
Write-Host "Output saved to: $targetPath"
Write-Host "Total lines: $lineCount"
Write-Host "Total words: $wordCount"
