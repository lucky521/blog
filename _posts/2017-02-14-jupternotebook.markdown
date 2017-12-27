
# This is jupyter notebook


jupyter notebook 将python和markdown结合在同一个文本文件。ipynb文件在呈现的时候既会按markdown格式呈现出文字段落、python代码，同时又会按顺序试试执行代码，将print的结果输出出来。


```python
import sys
print sys.version
```

    2.7.10 (default, Feb  7 2017, 00:08:15)
    [GCC 4.2.1 Compatible Apple LLVM 8.0.0 (clang-800.0.34)]



暂时我还没有办法在github jekyll 直接解析ipynb文件。而需要事先使用命令将ipynb转换为markdown文件。

  $ jupyter nbconvert hello.ipynb --to markdown
