@ECHO OFF


:MENU
CLS

TYPE "bin\menu.txt" && (
REM
) || (
PAUSE
EXIT
)

SET /P choice="Choice>>"

IF %choice% EQU 1 (
GOTO ONE
)
IF %choice% EQU 2 (
GOTO TWO
)
IF %choice% EQU 3 (
GOTO THREE
)
IF %choice% EQU 4 (
GOTO FOUR  
)
IF %choice% EQU 5 (
GOTO FIVE    
)
IF %choice% EQU 6 (
GOTO SIX
)


GOTO MENUERROR 


:ONE
IF EXIST bin\trainingData.npy (
    ECHO Data has already been created!
    PAUSE
    GOTO MENU
)
ECHO Creating data...
CALL bin\dataCreator\dist\dataCreator.exe


:IFFILEEXIST
IF EXIST bin\trainingData.npy (
    PAUSE
    GOTO MENU
)
GOTO IFFILEEXIST


:TWO
IF EXIST bin\model.pth (
    ECHO The CNN has been already trained!
    PAUSE
    GOTO MENU
)
IF NOT EXIST bin\trainingData.npy (
    ECHO There's no dataset!
    PAUSE
    GOTO MENU
)

ECHO Training the cnn...
CALL C:\ProgramData\Anaconda3\Scripts\activate.bat
python main.py
ECHO Training completed!
PAUSE


GOTO MENU
:THREE
IF NOT EXIST bin\model.pth (
    GOTO NOTTRAINEDERROR
)

SET /P fileChoice="Enter the name of the file>>"

IF EXIST bin\userdata\%fileChoice% (
    CALL C:\ProgramData\Anaconda3\Scripts\activate.bat
    python getResult.py %1 %fileChoice%
    PAUSE
    GOTO MENU
)

GOTO FILEERROR


GOTO MENU
:FOUR
CLS
TYPE "bin\info.txt" && (
PAUSE
GOTO MENU
) || (
PAUSE
GOTO MENU
)

:FIVE
EXIT

:SIX
MKDIR backup
XCOPY getResult.py backup
XCOPY main.py backup
XCOPY Net.py backup
XCOPY bin\dataCreator\dataCreator.py backup
GOTO MENU

:MENUERROR
ECHO Menu Error: Incorrect choice!
PAUSE
GOTO MENU

:FILEERROR
ECHO File Error: File doesn't exist!
PAUSE
GOTO MENU

:NOTTRAINEDERROR
ECHO Module Error: The CNN isn't trained!
PAUSE
GOTO MENU

ECHO ON