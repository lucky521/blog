---
title: "经常反复使用的代码片段"
subtitle: ""
categories: [productivity]
layout: post
---


## Hive

###
```shell
hive -e "set hive.cli.print.header=false;show partitions app.app_query_attr_feature;" | tail -1 | cut -d'=' -f2
```