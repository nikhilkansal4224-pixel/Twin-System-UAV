!define APP_NAME "UAV Digital Twin Ground Control Station"
!define COMP_NAME "UAV Aero Systems"
!define VERSION "1.0.0"
!define EXE_NAME "UAV_Digital_Twin.exe"

Name "${APP_NAME}"
OutFile "dist\UAV_Digital_Twin_Setup_v1.0.exe"
InstallDir "$PROGRAMFILES64\UAV_Digital_Twin"
RequestExecutionLevel admin

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section "MainSection" SEC01
    SetOutPath "$INSTDIR"
    
    # Copy all compiled PyInstaller dist files
    File /r "dist\UAV_Digital_Twin\*.*"
    
    # Create Desktop Shortcut
    CreateShortCut "$DESKTOP\UAV Digital Twin.lnk" "$INSTDIR\${EXE_NAME}" "" "$INSTDIR\assets\app_icon.ico"
    
    # Create Start Menu Shortcuts
    CreateDirectory "$SMPROGRAMS\UAV Digital Twin"
    CreateShortCut "$SMPROGRAMS\UAV Digital Twin\UAV Digital Twin.lnk" "$INSTDIR\${EXE_NAME}"
    CreateShortCut "$SMPROGRAMS\UAV Digital Twin\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
    
    # Write Uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$DESKTOP\UAV Digital Twin.lnk"
    RMDir /r "$SMPROGRAMS\UAV Digital Twin"
    RMDir /r "$INSTDIR"
SectionEnd