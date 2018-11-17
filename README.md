blog
====

Lucky Blog at Github

[Click me][1] to Archive List Page.

[1]:https://lucky521.github.io/blog



## How to build it

Jekyll is a static site generator. It is like a file-based CMS, but it's not a CMS.

### Prerequisite for ubuntu
```
$ sudo apt-get install python-software-properties
$ sudo apt-add-repository ppa:brightbox/ruby-ng
$ sudo apt-get update
$ sudo apt-get install ruby2.1 ruby-switch
$ sudo ruby-switch --set ruby2.1
$ sudo apt-get install ruby2.1 ruby2.1-dev make gcc nodejs
$ sudo gem install jekyll
$ sudo gem install github-pages
```

### Run it in project folder
```
$ cd blog
$ jekyll serve --host=0.0.0.0
```


### Feature Support

Math formula Support : Mathjax