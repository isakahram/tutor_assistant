@echo off
chcp 65001 >nul
title Расшифровка занятий
cd /d "%~dp0"

echo Проверяю Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Python не найден.
    echo Установите его с https://www.python.org/downloads/
    echo При установке обязательно отметьте "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Первый запуск: создаю виртуальное окружение и ставлю библиотеки.
    echo Это займёт несколько минут, нужен интернет.
    python -m venv venv
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось создать окружение venv.
        pause
        exit /b 1
    )
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось установить зависимости.
        pause
        exit /b 1
    )
)

venv\Scripts\python.exe main.py
if errorlevel 1 pause
