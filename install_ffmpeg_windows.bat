@echo off
echo Installing FFmpeg for Windows...
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as administrator
) else (
    echo Please run this script as administrator!
    pause
    exit /b 1
)

REM Download FFmpeg
echo Downloading FFmpeg...
powershell -Command "& {Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'ffmpeg.zip'}"

REM Extract FFmpeg
echo Extracting FFmpeg...
powershell -Command "& {Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'ffmpeg_temp' -Force}"

REM Find the extracted folder
for /d %%i in (ffmpeg_temp\*) do set FFMPEG_DIR=%%i

REM Copy to Program Files
echo Installing to C:\ffmpeg...
xcopy "%FFMPEG_DIR%\bin\*" "C:\ffmpeg\bin\" /E /I /H /Y
xcopy "%FFMPEG_DIR%\doc\*" "C:\ffmpeg\doc\" /E /I /H /Y
xcopy "%FFMPEG_DIR%\presets\*" "C:\ffmpeg\presets\" /E /I /H /Y

REM Add to PATH
echo Adding FFmpeg to PATH...
setx /M PATH "%PATH%;C:\ffmpeg\bin"

REM Clean up
echo Cleaning up...
rd /s /q ffmpeg_temp
del ffmpeg.zip

echo.
echo FFmpeg installation complete!
echo Please restart your command prompt and verify with: ffmpeg -version
pause