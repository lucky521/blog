---
title: "用MiniDumpWriteDump生成指定进程的dump文件"
subtitle: "MiniDumpWriteDump "
categories: [design]
layout: post
---

在一个独立的程序里，根据特定进程名称，来生成它的dump。

```
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









<!--
这里是注释区

```
print "hello"
```

***Stronger***

![My image]({{ site.baseurl }}/images/emule.png)

My Github is [here][mygithub].
[mygithub]: https://github.com/lucky521

-->