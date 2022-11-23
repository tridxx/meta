from typing import Union, Any, Generator

import jieba

seglist: Generator[Union[str, Any], Any, None] = list(jieba.cut("小明1995年毕业于清华大学",cut_all=False))
print(seglist)