---
layout: default
title: 愤怒的傻鸟
---

[AngryBirdGame](https://github.com/lucky521/Pig-Hate-Angry-Bird)
=============

This is a Flying Game about AngryBird.


This game is writtern by Python base on Pyglet.
To learn more about Pyglet, you can refer to http://www.pyglet.org/



The source code structure:  
.   
|- luckybird.py   the main proceture    
|-  game    
&emsp;|- _ init _.py   modules used by luckybird.py   
&emsp;|- config.py     configure file, global constant   
&emsp;|- resources.py    import resources   
&emsp;|- util.py     util functions   
&emsp;|- player.py   Player class is about the player himself, derived from PhysicalObject   
&emsp;|- angrybird.py    AngryBird class is about the enemy, derived from PhysicalObject   
&emsp;|- egg.py      Egg class is about the score, derived from PhysicalObject   
&emsp;|- physicalobject.py   Base class, moving things   
&emsp;|- load.py   instantiate and load all objects   

### [How to Play](https://github.com/lucky521/Pig-Hate-Angry-Bird/blob/master/readme.txt)
