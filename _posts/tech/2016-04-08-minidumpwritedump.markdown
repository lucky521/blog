---
title: "用MiniDumpWriteDump生成指定进程的dump文件"
subtitle: "MiniDumpWriteDump "
categories: [Tech]
layout: post
---
# Dump的需求

在一个独立的程序里，根据特定进程名称，来生成它的dump。

# Snippet

先用CreateToolhelp32Snapshot获取所有进程的snapshot，然后Process32Next进行遍历，比较每一个进程的名字找出目标名称，OpenProcess获取句柄，然后MiniDumpWriteDump抓取到自建的文件里面。

```cpp
#include <windows.h>
#include <DbgHelp.h>
#include <tlhelp32.h>
#pragma comment(lib, "dbghelp.lib")

PROCESSENTRY32 entry;
HANDLE snapshot, mkshProcess, file_handle;
MINIDUMP_TYPE flags;
uint32 dump_rv;

entry.dwSize = sizeof(PROCESSENTRY32);
snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, (DWORD)0);
if (Process32First(snapshot, &entry) == TRUE)
{
    while (Process32Next(snapshot, &entry) == TRUE)
    {
        if (_tcsicmp(entry.szExeFile, _T("target-name.exe")) == 0)
        {  
            mkshProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, entry.th32ProcessID);
            file_handle = CreateFile(TEXT("C:\\lu-dump-test.dmp"), GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL ); 
            flags = (MINIDUMP_TYPE)(MiniDumpWithFullMemory | MiniDumpWithFullMemoryInfo | MiniDumpWithHandleData | MiniDumpWithUnloadedModules | MiniDumpWithThreadInfo);
            Warning("Newlog: mkshandle=%u, mksPid=%u, filehandle=%u.\n", mkshProcess, entry.th32ProcessID, file_handle);
            dump_rv = MiniDumpWriteDump( mkshProcess, entry.th32ProcessID, file_handle, flags,  0, 0, 0 );
            if(!dump_rv) 
                Warning("Newlog: MiniDumpWriteDump failed %u\n",GetLastError() );
            CloseHandle(mkshProcess);
        }
    }
}
CloseHandle(snapshot);
 
```

这段代码在有些场合下不知道为什么，MiniDumpWriteDump会调用失败，GetLastError错误码为2147942699(0x8007012B)。


# 写成一个独立的程序

读入命令行参数来设置生成的dump文件的名称。在外部调用的时候可以这样：

`ShellExecute(0,"open","C:\\collect_dump_of_remotemks.exe","remotemks-dump.dmp","",SW_SHOWNORMAL);`



```cpp
#include "stdafx.h"
#include <windows.h>
#include <DbgHelp.h>
#include <tlhelp32.h>
#include <time.h>
#include <stdio.h>
#pragma comment(lib, "dbghelp.lib")

int _tmain(int argc, _TCHAR* argv[])
{
    PROCESSENTRY32 entry;
    entry.dwSize = sizeof(PROCESSENTRY32);
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, (DWORD)0);

    if (Process32First(snapshot, &entry) == TRUE)
    {
        while (Process32Next(snapshot, &entry) == TRUE)
        {
			if (_tcsicmp(entry.szExeFile, _T("target.exe")) == 0)
            {  
                HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, entry.th32ProcessID);

				printf("hprocess:%08x\n", hProcess);
				printf("entry.th32ProcessID:%d\n", entry.th32ProcessID);

				wchar_t dumpfilename[200];
				swprintf_s(dumpfilename,200,  L"C:\\%s", argv[1]);
				HANDLE file_handle = CreateFile((LPCWSTR)dumpfilename, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL ); 
				MINIDUMP_TYPE flags = (MINIDUMP_TYPE)(MiniDumpWithFullMemory | MiniDumpWithFullMemoryInfo | MiniDumpWithHandleData | MiniDumpWithUnloadedModules | MiniDumpWithThreadInfo);
				int rv = MiniDumpWriteDump( hProcess, entry.th32ProcessID, file_handle, flags,  0, 0, 0 );
				printf("result of MiniDumpWriteDump=%d.\nfilename=%s.\n",rv, dumpfilename);

                CloseHandle(hProcess);
            }
        }
    }

    CloseHandle(snapshot);
    printf("collect-dump: Done.\n");
    return 0;
}
```



